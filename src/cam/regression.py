import cv2
import random

import numpy as np

__author__ = 'Kohistan'


def merge(segments):
    merged = []
    discarded = []

    while 1 < len(segments):
        #i = random.randint(0, len(segments) - 2)
        i = len(segments) - 2
        seg = segments.pop(i)
        neighb = segments[i]
        #while abs(neighb.intercept - seg.intercept) < 10:

        p1 = seg.coords[0:2]
        p2 = seg.coords[2:4]
        p3 = neighb.coords[0:2]
        p4 = neighb.coords[2:4]
        ndarray = np.vstack([p1, p2, p3, p4])
        points = np.float32(ndarray)

        regression = cv2.fitLine(points, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        error = _error(points, regression)

    return merged


def _error(points, regr):

    vx = regr[0][0]
    vy = regr[1][0]
    x0 = regr[2][0]
    y0 = regr[3][0]

    # column vectors for matrix calculation
    vect = np.vstack([vx, vy])
    p0 = np.vstack([x0, y0])

    # todo remove normalization because they are already
    projector = vect.dot(vect.T) / vect.T.dot(vect)  # projection matrix

    error = 0
    for point in points:
        actual = np.vstack([point[0], point[1]])  # make sure we have column vector here as well
        regressed = projector.dot(actual - p0) + p0

        print "regressed\n" + str(regressed)
        print "actual\n" + str(actual)

        errvect = actual - regressed
        err = errvect.T.dot(errvect)
        print "error: " + str(err) + "\n"

        error += err

    return error

if __name__ == '__main__':
    points = [(3.0, 0.0)]
    regression = [[2.0], [1.0], [0.0], [0.0]]
    _error(points, regression)

































