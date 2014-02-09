from bisect import insort
from time import time
from numpy import float32, array, vstack
import cv2

from config.cvconf import canonical_size as csize
from core.imgutil import draw_circles, draw_lines, order_hull
from core.video import VidProcessor

__author__ = 'Arnaud Peloquin'


class BoardFinder(VidProcessor):
    """
    Abstract class providing a suggestion of common features that should be shared by board-finding algorithms.

    The core requirement is to hold a self.mtx object when detection has been successful,
    expected to be the matrix used in perspective transforms.

    """

    def __init__(self, vmanager):
        super(BoardFinder, self).__init__(vmanager)
        self.corners = GobanCorners()
        self.mtx = None
        self.last_positive = -1.0  # last time the board was detected

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

    def _detect(self, frame):
        """
        Process the provided frame to find goban corners in it, and update the field "self.corners".
        Return True to indicate that the Goban has been located successfully (all 4 corners have been located).

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def perform_undo(self):
        self.mtx = None
        self.undoflag = False


class GobanCorners():
    """
    Data structure representing the corners of the Goban in an image, having some basic checking
    and drawing abilities.

    """
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

    def clear(self):
        self._points = []
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

    def _check(self):
        """
        If self.points forms a convex hull of 4 vertices, store these into self.hull
        Otherwise erase self.hull

        """
        if 3 < len(self._points):
            cvhull = cv2.convexHull(vstack(self._points))
            self.hull = order_hull([x[0] for x in cvhull])
            if len(self.hull) != 4:
                self.hull = None
        else:
            self.hull = None

    def __str__(self):
        return "Corners:" + str(self._points)


class SegGrid:
    """
    A structure that store line segments in two different categories: horizontal and vertical.
    This implementation is not good as it should instead make two categories based on their
    reciprocal orthogonality.

    These two groups can represent the horizontal and vertical lines of a goban.

    """
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


class SegGridIter(object):
    """
    Iterator used in SegGrid.__iter__()

    """
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
            assert self.idx == len(self.grid), "Should describe entire grid once and only once."
            raise StopIteration