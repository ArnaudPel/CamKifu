import math
from bisect import insort
import random
import os

import cv2
from cv import CV_CAP_PROP_POS_AVI_RATIO as POS_RATIO
import numpy as np
from board.boardbase import BoardFinder

from core.imgutil import split_sq, Segment
from config.devconf import gobanloc_npz


__author__ = 'Kohistan'


def hough(gray, prepare=True):
    """
    Calls the Canny function in order to call HoughLinesP function
    Returns all lines found.
    """
    # todo extract parameters in config file to ease tuning ?
    # todo remove "prepare" named parameter if not used
    if prepare:
        #prepd = cv2.GaussianBlur(gray, (3, 3), 0)
        #show(gray, name="gray", loc=(250, 250))
        #prepd = cv2.erode(gray, np.ones((1, 5)), iterations=4)
        #show(prepd, name="eroded", loc=(500, 250))
        #cv2.waitKey()
        prepd = gray
        prepd = cv2.Canny(prepd, 30, 90)
    else:
        prepd = gray

    threshold = 10
    #minlen = 3*min(bw.shape) / 4  # minimum length to accept a line
    minlen = min(prepd.shape) / 2  # minimum length to accept a line
    maxgap = minlen / 14  # maximum gap for line merges (if probabilistic hough)

    lines = cv2.HoughLinesP(prepd, 1, math.pi / 180, threshold, minLineLength=minlen, maxLineGap=maxgap)
    if lines is not None:
        return lines[0]
    else:
        return []


def find_segments(img, nbsplits=5):
    """
    Split the image into squares, call Hough on each square.
    Returns all segments found in two lists. The first contains the segments deemed "horizontal",
    and the other segment contains the rest, the "verticals". Both are ordered by intercept,
    their intersection with one of the middle line of the image.

    """

    hsegs = []
    vsegs = []
    chunks = list(split_sq(img, nbsplits=nbsplits))
    #chunks.extend(split_sq(img, nbsplits=10, offset=True))  # todo connect that if needing more segments
    for chunk in chunks:
        segments = hough(chunk.mat)
        for seg in segments:
            # translate segment coordinates to place it in global image
            seg[0] += chunk.x
            seg[1] += chunk.y
            seg[2] += chunk.x
            seg[3] += chunk.y
            segment = Segment(seg, img)
            if segment.horiz:
                insort(hsegs, segment)
            else:
                insort(vsegs, segment)

    return SegGrid(hsegs, vsegs, img)


if __name__ == '__main__':
    src = np.ones((5, 5), np.uint8)
    sub = src[1:4, 1:4]
    print(src)
    for x in range(sub.shape[1]):
        for y in range(sub.shape[0]):
            sub[x][y] = 0
    print(src)
    print zip(range(sub.shape[1]), range(sub.shape[0]))


class SegGridIter(object):
    def __init__(self, grid):
        self.grid = grid
        self.idx = -1

    def __iter__(self):
        return self

    def next(self):
        self.idx += 1
        l1 = len(self.grid.hsegs)
        if self.idx < l1:
            return self.grid.hsegs[self.idx]
        elif self.idx - l1 < len(self.grid.vsegs):
            return self.grid.vsegs[self.idx - l1]
        else:
            assert self.idx == len(self.grid), "Should describe entire grid once and only once."  # todo remove
            raise StopIteration


class SegGrid:
    def __init__(self, hsegs, vsegs, img):
        assert isinstance(hsegs, list), "hsegs must be of type list."
        assert isinstance(vsegs, list), "vsegs must be of type list."
        self.hsegs = hsegs
        self.vsegs = vsegs
        self.img = img

    def __add__(self, other):
        assert isinstance(other, SegGrid), "can't add: other should be a grid."
        assert self.img.shape == other.img.shape, "images should have same shape when adding grids."
        hsegs = [seg for seg in self.hsegs + other.hsegs]
        vsegs = [seg for seg in self.vsegs + other.vsegs]
        hsegs.sort()
        vsegs.sort()
        return SegGrid(hsegs, vsegs, self.img)

    def __iter__(self):
        return SegGridIter(self)

    def __len__(self):
        return len(self.hsegs) + len(self.vsegs)

    def __str__(self):
        rep = "Grid(hsegs:" + str(len(self.hsegs))
        rep += ", vsegs:" + str(len(self.vsegs)) + ")"
        return rep

    def enumerate(self):
        return self.hsegs + self.vsegs

    def insort(self, segment):
        insort(self.hsegs, segment) if segment.horiz else insort(self.vsegs, segment)


def runmerge(grid):
    merged = SegGrid([], [], grid.img)
    discarded = SegGrid([], [], grid.img)

    # merge locally first to minimize askew lines
    for i in range(2):  # i=0 is horizontal, i=1 is vertical
        low = SegGrid([], [], grid.img)
        high = SegGrid([], [], grid.img)
        mid = grid.img.shape[1 - i] / 2  # numpy (i,j) is equivalent to opencv (y,x)
        for seg in (grid.vsegs if i else grid.hsegs):
            if (seg.coords[0 + i] + seg.coords[2 + i]) / 2 < mid:
                low.insort(seg)
            else:
                high.insort(seg)
        mgd, disc = _merge(low)
        merged += mgd
        discarded += disc
        mgd, disc = _merge(high)
        merged += mgd
        discarded += disc

    # run global merges with increasing tolerance for error
    merged += discarded
    for precision in (1, 4, 16, 32):
        #print "nb segments: " + str(len(merged))
        #print "precision:" + str(precision)
        #print
        merged, discarded = _merge(merged, precision=precision)
        merged += discarded
    return merged


def _merge(grid, precision=1):
    """
    Returns:
        merged_grid, discarded_grid
    Args:
        precision -- the max "dispersion" allowed around the regressed line to accept a merge.

    """

    merged = SegGrid([], [], grid.img)
    discarded = SegGrid([], [], grid.img)
    random.seed(42)
    correction = 0

    for i in range(2):
        if i:
            segments = list(grid.vsegs)  # second pass, vertical lines
        else:
            segments = list(grid.hsegs)  # first pass, horizontal lines

        while 1 < len(segments):
            i = random.randint(0, len(segments) - 1)
            seg = segments.pop(i)
            valuations = []

            # valuate against the 'n' closest neighbours
            for neighb in _get_neighbours(segments, i, seg.intercept):
                _least_squares(seg, neighb, valuations)
                if 0 < len(valuations) and valuations[0][0] < 0.1:  # error small enough already, don't loop
                    break

            if 0 < len(valuations):
                bestmatch = valuations[0]
                if bestmatch[0] < precision:  # if error acceptable
                    segments.remove(bestmatch[1])
                    segmt = Segment(_get_seg(bestmatch[2]), grid.img)
                    merged.insort(segmt)
                else:
                    discarded.insort(seg)
            else:
                discarded.insort(seg)

                #print "seg: " + str(seg.coords)
                #print "error: " + str(bestmatch[0])
                #print "neigh: " + str(bestmatch[1].coords)
                #print "projections: "
                #for proj in bestmatch[2]:
                #    print proj
                #print

        # last segment has not been merged, but is not necessarily bad either
        if 0 < len(segments):
            merged.insort(segments[0])
            correction -= 1  # -1 because last segment of this pass has not been merged
    assert len(grid) == 2 * len(merged) + len(discarded) + correction
    return merged, discarded


def _least_squares(seg, neighb, valuations):
    p1 = seg.coords[0:2]
    p2 = seg.coords[2:4]
    p3 = neighb.coords[0:2]
    p4 = neighb.coords[2:4]
    ndarray = np.vstack([p1, p2, p3, p4])
    points = np.float32(ndarray)
    regression = cv2.fitLine(points, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
    error, projections = _error(points, regression)
    insort(valuations, (error, neighb, projections))


def _get_neighbours(segments, start, intercept):
    """
    Generator that returns segments whose intercept is increasingly more distant from
        the "intercept" argument. The list is supposed to be sorted, and "start" is supposed
        to be the position of "intercept" in the list.

    """
    l = start - 1
    r = start
    stop = len(segments)
    while 0 <= l or r < stop:

        # get neighbours in both directions
        if 0 <= l:
            left = segments[l]
        else:
            left = None
        if r < stop:
            right = segments[r]
        else:
            right = None

        # select best direction
        if right is None:
            yield left
            l -= 1
            continue
        if left is None:
            yield right
            r += 1
            continue
        ldiff = intercept - left.intercept
        rdiff = right.intercept - intercept

        # todo remove dev check
        if ldiff < 0 or rdiff < 0:
            raise IndexError("There is an issue with left / right ordering")

        if 10 < min(ldiff, rdiff):  # todo tune pixel dist according to img.shape ?
            return # don't explore neighbours whose intercept is more than 'n' pixels away

        if ldiff < rdiff:
            yield left
            l -= 1
        else:
            yield right
            r += 1


def _error(points, regr):
    """
    >>> points = [(1.0, 0.0)]
    >>> regression = [[0.7071067811865475], [0.7071067811865475], [0.0], [0.0]]
    >>> print _error(points, regression)
    (0.5, [[0.5, 0.5]])

    >>> points = [(-4.0, 5.0)]
    >>> regression = [[1.0], [5.0/3], [0.0], [-5.0]]
    >>> print _error(points, regression)
    (73.529411764705884, [[3.3529411764705888, 0.58823529411764763]])

    """
    vx = regr[0][0]
    vy = regr[1][0]
    x0 = regr[2][0]
    y0 = regr[3][0]
    # column vectors for matrix calculation
    vect = np.vstack([vx, vy])
    p0 = np.vstack([x0, y0])

    projector = vect.dot(vect.T) / vect.T.dot(vect)  # projection matrix

    error = 0
    projections = []
    for point in points:
        actual = np.vstack([point[0], point[1]])  # make sure we have column vector here as well
        projection = projector.dot(actual - p0) + p0
        errvect = actual - projection
        err = errvect.T.dot(errvect)
        error += err
        projections.append([c[0] for c in projection])
    return error[0][0], projections


def _get_seg(points):
    """
    return the two most distant points from each other of the given list.
    note: because the expected number of points is 4 in this context, the most
     basic comparison has been used, in o(n2)

    points -- any collection of (x,y) couples having [] defined should work.

    >>> points = [(2, 0), (0, 1), (-1, 0), (0, -1)]
    >>> print _get_seg(points)
    (-1, 0, 2, 0)

    """
    if len(points) < 3:
        return points

    seg = None
    dist = 0
    for i in range(1, len(points)):
        p0 = points[i]
        # todo if we assume the points are roughly aligned, no need for the double loop
        for j in range(i):
            p1 = points[j]
            curdist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
            if dist < curdist:
                seg = (p0[0], p0[1], p1[0], p1[1])
                dist = curdist
    return seg


class BoardFinderManual(BoardFinder):

    def __init__(self, vmanager, rectifier):
        """
        self.capture_pos -- used to lock the position of cvCapture when waiting for user to locate goban.
                            has no impact on live cam, but prevent unwanted frames consumption for files.

        """
        super(BoardFinderManual, self).__init__(vmanager, rectifier)
        self.name = "Manual Grid Detection"
        self.manual_found = False
        self.capture_pos = None
        try:
            np_file = np.load(gobanloc_npz)
            for p in np_file["location"]:
                self.corners.add(p)
            self.manual_found = True
        except IOError or TypeError:
            pass

    def _detect(self, frame):
        if self.undoflag:
            self.perform_undo()
        if not self.manual_found:
            self._lockpos()
            detected = False
        else:
            self._standby()
            detected = True
        self.corners.paint(frame)
        self._show(frame, name=self.name)
        return detected

    def _lockpos(self):
        if self.capture_pos is None:
            self.capture_pos = self.vmanager.capt.get(POS_RATIO)
        else:
            self.vmanager.capt.set(POS_RATIO, self.capture_pos)
        cv2.setMouseCallback(self.name, self.onmouse)

    def _standby(self):
        self.capture_pos = None

    #noinspection PyUnusedLocal
    def onmouse(self, event, x, y, flag, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not self.corners.ready():
            self.corners.add((x, y))
            if self.corners.ready():
                self.manual_found = True
                # todo comment that before publish
                np.savez(gobanloc_npz, location=self.corners._points)

    def perform_undo(self):
        super(BoardFinderManual, self).perform_undo()
        self.manual_found = False
        self.corners.pop()
        try:
            os.remove(gobanloc_npz)
        except OSError:
            pass




















