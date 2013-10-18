import math
from bisect import insort
import random
import sys

import cv2
import numpy as np
import os

from cam.imgutil import split_sq, Segment, VidProcessor, show, draw_circles, draw_lines
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
        bw = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.Canny(bw, 10, 100)
    else:
        bw = gray

    threshold = 10
    #minlen = 3*min(bw.shape) / 4  # minimum length to accept a line
    minlen = min(bw.shape) / 2  # minimum length to accept a line
    maxgap = minlen / 12  # maximum gap for line merges (if probabilistic hough)

    lines = cv2.HoughLinesP(bw, 1, math.pi / 180, threshold, minLineLength=minlen, maxLineGap=maxgap)
    if lines is not None:
        return lines[0]
    else:
        return []


def find_segments(img):
    """
    Split the image into squares, call Hough on each square.
    Returns all segments found in two lists. The first contains the segments deemed "horizontal",
    and the other segment contains the rest, the "verticals". Both are ordered by intercept,
    their intersection with one of the middle line of the image.

    """

    hsegs = []
    vsegs = []
    chunks = list(split_sq(img, nbsplits=10))
    #chunks.extend(split_sq(img, nbsplits=10, offset=True))  # todo connect that if needing more segments
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        i += 1
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

    return Grid(hsegs, vsegs, img)


if __name__ == '__main__':
    src = np.ones((5, 5), np.uint8)
    sub = src[1:4, 1:4]
    print(src)
    for x in range(sub.shape[1]):
        for y in range(sub.shape[0]):
            sub[x][y] = 0
    print(src)
    print zip(range(sub.shape[1]), range(sub.shape[0]))


class Grid:
    def __init__(self, hsegs, vsegs, img):
        assert isinstance(hsegs, list), "hsegs must be of type list."
        assert isinstance(vsegs, list), "vsegs must be of type list."
        self.hsegs = hsegs
        self.vsegs = vsegs
        self.img = img

    def __add__(self, other):
        assert isinstance(other, Grid), "can't add: other should be a grid."
        assert self.img.shape == other.img.shape, "images should have same shape when adding grids."
        hsegs = [seg for seg in self.hsegs + other.hsegs]
        vsegs = [seg for seg in self.vsegs + other.vsegs]
        hsegs.sort()
        vsegs.sort()
        return Grid(hsegs, vsegs, self.img)

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
    merged = Grid([], [], grid.img)
    discarded = Grid([], [], grid.img)

    # merge locally first to minimize askew lines
    for i in range(2):  # i=0 is horizontal, i=1 is vertical
        low = Grid([], [], grid.img)
        high = Grid([], [], grid.img)
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

    merged = Grid([], [], grid.img)
    discarded = Grid([], [], grid.img)
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


class BoardFinder(VidProcessor):
    def __init__(self, camera, rectifier):
        super(self.__class__, self).__init__(camera, rectifier)
        self.name = "Manual Grid Detection"
        self.corners = GridListener()
        self.mtx = None
        self.size = 19 * 25

    def _doframe(self, frame):
        cv2.setMouseCallback(self.name, self.corners.onmouse)
        if self.undo:
            self.perform_undo()
        elif self.corners.ready():
            source = np.array(self.corners.hull, dtype=np.float32)
            dst = np.array([(0, 0), (self.size, 0), (self.size, self.size), (0, self.size)], dtype=np.float32)
            # todo optimization: crop the image around the ROI before computing the transform
            try:
                self.mtx = cv2.getPerspectiveTransform(source, dst)
                self._done()
            except cv2.error:
                print "Please mark a square-like area. The 4 points must form a convex hull."
                self.undo = True
        self.corners.paint(frame)
        show(frame, name=self.name)

    def perform_undo(self):
        self.corners.undo()
        self.mtx = None
        try:
            os.remove(gobanloc_npz)
        except OSError:
            pass
        self.undo = False


class GridListener():
    def __init__(self, nb=4):
        self.nb = nb
        self.hull = None
        try:
            np_file = np.load(gobanloc_npz)
            self.points = [p for p in np_file["location"]]
            self._order()
        except IOError or TypeError:
            self.points = []

    #noinspection PyUnusedLocal
    def onmouse(self, event, x, y, flag, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not self.ready():
            self.points.append((x, y))
            if self.ready():
                self._order()

    def undo(self):
        if len(self.points):
            self.points.pop(-1)

    def ready(self):
        return len(self.points) == self.nb

    def paint(self, img):
        #draw the clicks
        draw_circles(img, self.points)

        #draw convex hull
        if self.ready():
            nbpts = len(self.hull) - 1
            color = (0, 0, 255)
            for i in range(-1, nbpts):
                x1, y1 = self.hull[i]
                x2, y2 = self.hull[i + 1]
                draw_lines(img, [[x1, y1, x2, y2]], color)
                color = (255 * (nbpts - i - 1) / nbpts, 0, 255 * (i + 1) / nbpts)

                # draw extrapolated
                #if len(self.hull) == 4:
                #    segs = []
                #    for i in [-1, 0]:
                #        p11 = self.hull[i]
                #        p12 = self.hull[i + 1]
                #        p21 = self.hull[i + 2]
                #        p22 = self.hull[i + 3]
                #
                #size = 18
                #for j in range(1, size):
                #    x1 = (j * p11[0] + (size - j) * p12[0]) / size
                #    x2 = (j * p22[0] + (size - j) * p21[0]) / size
                #    y1 = (j * p11[1] + (size - j) * p12[1]) / size
                #    y2 = (j * p22[1] + (size - j) * p21[1]) / size
                #    segs.append([x1, y1, x2, y2])
                #draw_lines(img, segs, color=(42, 142, 42))

    def _order(self):
        self.hull = []
        idx = 0
        mind = sys.maxint
        if self.ready():
            cvhull = cv2.convexHull(np.vstack(self.points))
            for i in range(len(cvhull)):
                p = cvhull[i][0]
                dist = p[0] ** 2 + p[1] ** 2
                if dist < mind:
                    mind = dist
                    idx = i
            for i in range(idx, idx + len(cvhull)):
                p = cvhull[i % len(cvhull)][0]
                self.hull.append((p[0], p[1]))
            if len(self.hull) == 4:
                np.savez(gobanloc_npz, location=self.points)

    def __str__(self):
        return "Corners:" + str(self.points)