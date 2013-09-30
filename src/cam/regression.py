import cv2
import random

import numpy as np
from bisect import insort
from cam.draw import Segment
from cam.hough import Grid

__author__ = 'Kohistan'

itercount = 0


def runmerge(grid):
    merged = Grid([], [], grid.img)
    discarded = Grid([], [], grid.img)

    # merge locally first to minimize askew lines
    for i in range(2):  # i=0 is horizontal, i=1 is vertical
        low = Grid([], [], grid.img)
        high = Grid([], [], grid.img)
        mid = grid.img.shape[1-i] / 2  # numpy (i,j) is equivalent to opencv (y,x)
        for seg in (grid.vsegs if i else grid.hsegs):
            if (seg.coords[0+i] + seg.coords[2+i])/2 < mid:
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
    for precision in (1, 2, 4, 8, 16, 32):
        print "nb segments: " + str(len(merged))
        print "precision:" + str(precision)
        print
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
        if i: segments = list(grid.vsegs)  # second pass, vertical lines
        else: segments = list(grid.hsegs)  # first pass, horizontal lines

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
    assert len(grid) == 2*len(merged) + len(discarded) + correction
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

    global itercount
    itercount += 1


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
        for j in range(i):
            p1 = points[j]
            curdist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
            if dist < curdist:
                seg = (p0[0], p0[1], p1[0], p1[1])
                dist = curdist
    return seg


if __name__ == '__main__':
    pass






























