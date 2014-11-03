from bisect import insort
import random

import cv2
from math import sin, cos, pi
import numpy as np
from numpy import zeros, uint8, float32

from camkifu.board.boardfinder import BoardFinder, SegGrid
from camkifu.core.imgutil import Segment, sort_contours


__author__ = 'Arnaud Peloquin'


class BoardFinderAuto(BoardFinder):
    """
    Automatically detect the board location in an image. Implementation based on contours detection,
    with usage of median filtering. It is not able to detect the board when one or more stones are
    played on the first line, as those tend to break the contour.

    This implementation attempt is a work in progress.

    """

    label = "Automatic"

    def __init__(self, vmanager):
        super(BoardFinderAuto, self).__init__(vmanager)
        self.lines_accu = []
        self.groups_accu = []  # groups of points in the same image region
        self.passes = 0  # the number of frames processed since the beginning

    def _detect(self, frame):
        length_ref = min(frame.shape[0], frame.shape[1])  # a reference length linked to the image
        median = cv2.medianBlur(frame, 15)

        # todo instead of edge detection : threshold binarization on an HSV image to only retain, a certain color/hue ?
        #   the target color could be guessed from the global image :
        #   computing an histo of hues and keeping the most frequent, hoping the goban is taking enough space
        #   (maybe only try a mask reflecting the default projection of the goban rectangle on the image plane).
        canny = cv2.Canny(median, 25, 75)
        # first parameter is the input image (it seems). appeared in opencv 3.0-alpha and is missing from the docs
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        sorted_boxes = sort_contours(contours)
        biggest = sorted_boxes[-1]
        frame_area = frame.shape[0] * frame.shape[1]
        found = False
        if frame_area / 3 < biggest.box_area:
            segments = self.find_lines(biggest.pos, contours, length_ref, median.shape)
            self.lines_accu.extend(segments)
            # accumulate data of 4 images before running analysis
            if not self.passes % 4:
                self.group_intersections(length_ref)
                while 4 < len(self.groups_accu):
                    prev_length = len(self.groups_accu)
                    self.merge_groups(length_ref)
                    if len(self.groups_accu) == prev_length:
                        break  # seems it won't get any better
                found = self.updt_corners(median, length_ref)

        if not self.passes % 4:
            self.corners.paint(median)
            self.metadata.insert(0, "Board found" if found else "Looking for board..")
            self._show(median)
        self.passes += 1
        return found

    @staticmethod
    def find_lines(pos, contours, length_ref, shape):
        """
        Use houghlines to find the sides of the contour (the goban hopefully).

        Another method could have been cv2.approxPolyDP(), but it fits the polygon points inside the contour, whereas
        what is needed here is a nicely fitted "tangent" for each side: the intersection points (corners) are outside
        the contour most likely (round corners).

        -- pos: the position to use in the list of contours
        -- contours: the list of contours
        -- length_ref: a reference used to parametrize thresholds
        -- the shape of the image in which contours have been found

        """
        ghost = zeros((shape[0], shape[1]), dtype=uint8)
        cv2.drawContours(ghost, contours, pos, (255, 255, 255), thickness=1)
        thresh = int(length_ref / 5)
        lines = cv2.HoughLines(ghost, 1, pi / 180, threshold=thresh)
        segments = []
        for line in lines:
            rho, theta = line[0]
            a, b = cos(theta), sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
            pt2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
            segments.append(Segment((pt1[0], pt1[1], pt2[0], pt2[1]), ghost))
        # todo trim if there are too many lines ? merge them maybe ?
        return segments

    def group_intersections(self, length_ref):
        """
        Store tangent intersections into rough groups (connectivity-based clustering idea).
        The objective is to obtain 4 groups of intersections, providing estimators of the corners
        location.

        """
        for s1 in self.lines_accu:
            for s2 in self.lines_accu:
                if pi / 3 < s1.angle(s2):
                    assigned = False
                    p0 = s1.intersection(s2)
                    for g in self.groups_accu:
                        for p1 in g:
                            # if the two points are close enough, group them
                            if (p0[0] - p1[0]) ** 2 + (p0[0] - p1[0]) ** 2 < (length_ref / 80) ** 2:
                                g.append(p0)
                                assigned = True
                                break
                        if assigned:
                            break
                    if not assigned:
                        # create a new group for each lonely point, it may be joined with others later
                        self.groups_accu.append([p0])

    def merge_groups(self, length_ref):
        """
        Do one connectivity-based clustering pass on the current groups.
        This has been implemented quickly and is most likely not the best way to proceed.

        """
        todel = []
        toprint = []
        index = 0
        for g0 in self.groups_accu:
            merge = None
            for p0 in g0:
                for g1 in self.groups_accu:
                    if g0 is not g1 and g1 not in todel:
                        for p1 in g1:
                            if (p0[0] - p1[0]) ** 2 + (p0[0] - p1[0]) ** 2 < (length_ref / 50) ** 2:
                                merge = g1
                                break
                    if merge: break
                if merge: break
            if merge:
                merge.extend(g0)
                todel.append(g0)
                toprint.append(index)
            index += 1
        for gdel in todel:
            self.groups_accu.remove(gdel)
        if len(toprint):
            self.metadata.append("merged : g%s" % " g".join([str(x) for x in toprint]))

    def updt_corners(self, median, length_ref):
        """
        If 4 groups have been found: compute the mean point of each group and update corners accordingly.
        In any case clear accumulation structures to prepare for next detection round.

        """
        found = False
        nb_corners = 4
        if len(self.groups_accu) == nb_corners:
                # step 1 : compute mean centers (centers of gravity) of each group
                centers = []
                for group in self.groups_accu:
                    x, y = 0, 0
                    for pt in group:
                        x += pt[0]
                        y += pt[1]
                    centers.append((int(x / len(group)), int(y / len(group))))
                # step 2 : run some final checks on the resulting corners (only minimal side length for now)
                found = True
                for i in range(nb_corners):
                    segment = Segment((centers[i - 1][0], centers[i - 1][1], centers[i][0], centers[i][1]), median)
                    if segment.norm() < length_ref / 3:
                        found = False
                        break
                if found:
                    self.corners.clear()
                    for pt in centers:
                        self.corners.add(pt)
                        # cv2.circle(median, pt, 2, (50, 255, 255), thickness=2)
        self.metadata.append("clusters : %d" % len(self.groups_accu))
        # self.metadata.append("lines accum : %d" % len(self.lines_accu))
        # todo have a rolling cleanup over time ?
        self.lines_accu.clear()
        self.groups_accu.clear()
        return found

    def perform_undo(self):
        # todo does "undo" make any sense here ?
        super(BoardFinderAuto, self).perform_undo()
        self.corners.clear()

    def _window_name(self):
        return "Board Finder Auto"


def runmerge(grid):
    """
    Legacy function (sub-functions included), used here class mostly because they were at close hand
    at dev time. The 200ish lines of code taking the form of (_merge, _least_squares, _get_neighbours,
    _error, _get_seg) would most likely benefit from deletion and global rethink of the automated board_finder.

    """
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
        merged, discarded = _merge(merged, precision=precision)
        merged += discarded
    return merged


def _merge(grid, precision=1):
    """
    Merge segments that seem to appear to the same line. A merge between two segments is
    a least-square segment of the 4 points being merged, as per cv2.fitLine().

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

        # last segment has not been merged, but is not necessarily bad either
        if 0 < len(segments):
            merged.insort(segments[0])
            correction -= 1  # -1 because last segment of this pass has not been merged
    assert len(grid) == 2 * len(merged) + len(discarded) + correction
    return merged, discarded


def _least_squares(seg, neighb, valuations):
    """
    Merge "seg" and "neighb", and insert the resulting segment into "valuations",
    ordering on the regression error.

    """
    p1 = seg.coords[0:2]
    p2 = seg.coords[2:4]
    p3 = neighb.coords[0:2]
    p4 = neighb.coords[2:4]
    ndarray = np.vstack([p1, p2, p3, p4])
    points = np.float32(ndarray)
    regression = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
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

        if 10 < min(ldiff, rdiff):  # improvement: tune pixel dist according to img.shape ?
            return  # don't explore neighbours whose intercept is more than 'n' pixels away

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
    Return the two most distant points from each other of the given list.
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
        for j in range(i):
            p1 = points[j]
            curdist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
            if dist < curdist:
                seg = (p0[0], p0[1], p1[0], p1[1])
                dist = curdist
    return seg