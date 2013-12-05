from sys import maxint
from time import time
from numpy import float32, array, vstack
import cv2

from config.devconf import canonical_size as csize
from core.imgutil import draw_circles, draw_lines
from core.video import VidProcessor

__author__ = 'Kohistan'


class BoardFinder(VidProcessor):
    """
    Abstract class providing a suggestion of common features that should be
    shared by board-finding algorithms.

    The most inflexible part is to hold a self.mtx object when detection has been successful,
    expected to be the matrix used in perspective transforms.

    """

    def __init__(self, vmanager, rectifier):
        super(BoardFinder, self).__init__(vmanager, rectifier)
        self.corners = GobanCorners()
        self.mtx = None
        self.last_positive = -1.0  # never

    def _doframe(self, frame):
        if 3 < time() - self.last_positive:
            if self._detect(frame):
                source = array(self.corners.hull, dtype=float32)
                dst = array([(0, 0), (csize, 0), (csize, csize), (0, csize)], dtype=float32)
                try:
                    self.mtx = cv2.getPerspectiveTransform(source, dst)
                    self.last_positive = time()
                except cv2.error:
                    print "Please mark a square-like area. The 4 points must form a convex hull."
                    self.undoflag = True

    def perform_undo(self):
        self.mtx = None
        self.undoflag = False

    def _detect(self, frame):
        raise NotImplementedError("Abstract method meant to be extended")


class GobanCorners():
    def __init__(self, points=None):
        self.hull = None
        if points is not None:
            self._points = points
            self._check()
        else:
            self._points = []

    def ready(self):
        return self.hull is not None

    def add(self, point):
        self._points.append(point)
        self._check()

    def pop(self):
        if 0 < len(self._points):
            self._points.pop(-1)
            self._check()

    def paint(self, img):
        #draw points found so far
        draw_circles(img, self._points, color=(255, 255, 0))

        #draw convex hull
        if self.hull is not None:
            nbpts = len(self.hull) - 1
            color = (0, 0, 255)
            for i in range(-1, nbpts):
                x1, y1 = self.hull[i]
                x2, y2 = self.hull[i + 1]
                draw_lines(img, [[x1, y1, x2, y2]], color)
                color = (255 * (nbpts - i - 1) / nbpts, 0, 255 * (i + 1) / nbpts)

    def __str__(self):
        return "Corners:" + str(self._points)

    def _check(self):
        if len(self._points) == 4:
            self.hull = ordered_hull(self._points)
        else:
            self.hull = None

    def clear(self):
        self._points = []
        self._check()


def ordered_hull(points):
    hull = []
    idx = 0
    mind = maxint
    cvhull = cv2.convexHull(vstack(points))
    for i in range(len(cvhull)):
        p = cvhull[i][0]
        dist = p[0] ** 2 + p[1] ** 2
        if dist < mind:
            mind = dist
            idx = i
    for i in range(idx, idx + len(cvhull)):
        p = cvhull[i % len(cvhull)][0]
        hull.append((p[0], p[1]))
    return hull


