from time import time
import cv2
import sys

from numpy import float32, array

from camkifu.config.cvconf import canonical_size as csize, bf_loc
from camkifu.core.imgutil import draw_circles, draw_lines, get_ordered_hull, norm
from camkifu.core.video import VidProcessor


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
        last_positive = time() - self.last_positive
        if 10 < last_positive:  # re-run board location search 10s after last positive detection
            if self._detect(frame):
                source = array(self.corners.hull, dtype=float32)
                dst = array([(0, 0), (csize, 0), (csize, csize), (0, csize)], dtype=float32)
                try:
                    self.mtx = cv2.getPerspectiveTransform(source, dst)
                    self.last_positive = time()
                except cv2.error:
                    self.mtx = None  # the stones finder must stop
                    print("Please mark a square-like area. The 4 points must form a convex hull.")
        else:
            self.corners.paint(frame)
            self.metadata["Last detection {}s ago"] = int(last_positive)
            self._show(frame)

    def _detect(self, frame):
        """
        Process the provided frame to find goban corners in it, and update the field "self.corners".
        Return True to indicate that the Goban has been located successfully (all 4 corners have been located).

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_freq=2):
        """
        Override to take control of the location of the window of this boardfinder

        """
        super()._show(img, name, latency, thread, loc=bf_loc, max_frequ=max_freq)


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

    def is_ready(self):
        return self.hull is not None

    def submit(self, point):
        """
        Add the point if less than 4 have been provided, otherwise correct the closest point.

        """
        if len(self._points) < 4:
            self._points.append(point)
        else:
            closest = (sys.maxsize, None)  # (distance, index_in_list)
            for i, pt in enumerate(self._points):
                if norm(pt, point) < closest[0]:
                    closest = (norm(pt, point), i)
            self._points[closest[1]] = point
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
            self.hull = get_ordered_hull(self._points)
            if len(self.hull) != 4:
                self.hull = None
        else:
            self.hull = None

    def __str__(self):
        return "Corners:" + str(self._points)


