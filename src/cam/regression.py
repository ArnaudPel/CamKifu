import cv2
import random

import numpy as np
from bisect import insort
from cam.draw import Segment

__author__ = 'Kohistan'

itercount = 0


def merge(segments):
    """
    Returns:
        merged, discarded -- merged is a list of points, discarded a list of cam.draw.Segment

    """
    merged = []
    discarded = []
    random.seed(42)

    while 1 < len(segments):
        i = random.randint(0, len(segments) - 1)
        seg = segments.pop(i)
        valuations = []

        # valuate against the 'n' closest neighbours
        for neighb in _get_neighbours(segments, i, seg.intercept):
            _least_squares(seg, neighb, valuations)
            if 0 < len(valuations) and valuations[0][0] < 0.1:  # error small enough already, don't loop
                break

        if len(valuations) == 0:
            continue
        bestmatch = valuations[0]
        if bestmatch[0] < 1:  # if error acceptable
            segments.remove(bestmatch[1])  # todo allow segments to be merged several times by removing this line ?
            merged.append(_get_seg(bestmatch[2]))
        else:
            discarded.append(seg)

        #print "seg: " + str(seg.coords)
        #print "error: " + str(bestmatch[0])
        #print "neigh: " + str(bestmatch[1].coords)
        #print "projections: "
        #for proj in bestmatch[2]:
        #    print proj
        #print

    global itercount
    print "iterations: " + str(itercount)
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
    >>> segments = []
    >>> segments.append(Segment([], 0, 0))   # 15 - out
    >>> segments.append(Segment([], 0, 6))   # 9
    >>> segments.append(Segment([], 0, 9))   # 6
    >>> segments.append(Segment([], 0, 10))  # 5
    >>> segments.append(Segment([], 0, 13))  # 2
    >>> segments.append(Segment([], 0, 16))  # 1
    >>> segments.append(Segment([], 0, 18))  # 3
    >>> segments.append(Segment([], 0, 19))  # 4
    >>> segments.append(Segment([], 0, 30))  # 15 - out
    >>> start = 5
    >>> intercept = 15
    >>> diffs = [abs(neigh.intercept - intercept) for neigh in _get_neighbours(segments, start, intercept)]
    >>> print diffs
    [1, 2, 3, 4, 5, 6, 9]

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

        if 10 < min(ldiff, rdiff):
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

    # todo remove normalization if needing a bit more speed, because they are already
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






























