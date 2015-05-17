import traceback
import sys

import cv2
import numpy as np

import camkifu.core
from camkifu.core import imgutil
from camkifu.config import cvconf


class BoardFinder(camkifu.core.VidProcessor):
    """ Abstract class providing common features that may be used by board-finding algorithms.

    If a board finder is implemented without this base class, the core requirement is to maintain an 'mtx' attribute
    when detection has been successful. This 'mtx' is expected to be the matrix used in perspective transforms.

    Attributes:
        corners: GobanCorners
            Contains the corners that have been found so far.
        transform_dst: ndarray
            Provides the dimensions of the receiver array in perspective transform.
        mtx: ndarray
            The perspective transform matrix, that should be used by stones finder to straighten up the goban img.
    """

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.corners = GobanCorners()
        size = cvconf.canonical_size
        self.transform_dst = np.array([(0, 0), (size, 0), (size, size), (0, size)], dtype=np.float32)
        self.mtx = None

    def _doframe(self, frame):
        """ Analyse the image to detect a Goban board. If detected, update the perspective transform matrix accordingly.

        Args:
            frame: ndarray
                The image, potentially containing a Go board.
        """
        self.corners.frame = frame
        if self._detect(frame):
            source = np.array(self.corners.hull, dtype=np.float32)
            try:
                self.mtx = cv2.getPerspectiveTransform(source, self.transform_dst)
            except cv2.error:
                self.mtx = None  # the stones finder must stop
                traceback.print_exc()

    def _detect(self, frame):
        """ Process the provided frame to find goban corners in it, and update the 'corners' attribute.

        Returns detected: bool
            True to indicate that the Goban has been located successfully (all 4 corners have been located).
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_freq=2):
        """ Little method override, in order to take control of the location of the window of this boardfinder.
        """
        if loc is None:
            loc = cvconf.bf_loc
        super()._show(img, name, latency, thread, loc=loc, max_frequ=max_freq)


class GobanCorners():
    """ Data structure representing the corners of the Goban that have been found so far.
    Provides some basic checking and drawing abilities.

    Attributes:
        hull: list of points
            The convex hull of the detected points. Provides a basic checking of the consistency of those points.
            Eg. if the 4th point is inside the triangle formed by the 3 others, detection will fail. Thus this hull is
            what's actually used in the transform matrix computation.
        frame: ndarray
            The last frame processed. Used to get an idea of the sizes to expect.
        _points: list
            The corner points detected so far.
    """
    def __init__(self, points=None):
        self.hull = None
        self.frame = None
        if points is not None:
            self._points = points
            self._check_hull()
        else:
            self._points = []

    def is_ready(self):
        """ Return True if all 4 corners of the Goban have been found.
        """
        return self.hull is not None

    def submit(self, point):
        """ Append the point if less than 4 have been provided, otherwise correct the closest point.
        Note : in 'append' phase, a point too close to another point will be interpreted as an error and rejected.

        Args:
            point: (int, int)
                The point to append / update.
        """
        closest = (sys.maxsize, None)  # (distance, index_in_list)
        for i, pt in enumerate(self._points):
            if imgutil.norm(pt, point) < closest[0]:
                closest = (imgutil.norm(pt, point), i)
        if len(self._points) < 4:
            # security: don't append the provided point if it is too close to another:
            # the board is supposed to take a minimal amount of space in the image.
            if closest[1] is None or self.frame is None or min(*self.frame.shape[0:2]) / 5 < closest[0]:
                self._points.append(point)
        else:
            self._points[closest[1]] = point
        self._check_hull()

    def clear(self):
        """ Delete all the points registered so far, as well as the associated hull if any.
        """
        self._points = []
        self._check_hull()

    def paint(self, img):
        """ Draw corners points found so far on 'img', as well as their convex hull if it exists.
        """
        #
        for pt in self._points:
            cv2.circle(img, tuple(pt), 5, (255, 255, 0), thickness=1)
        if self.hull is not None:
            nbpts = len(self.hull) - 1
            color = (0, 0, 255)
            for i in range(-1, nbpts):
                x1, y1 = self.hull[i]
                x2, y2 = self.hull[i + 1]
                cv2.line(img, (x1, y1), (x2, y2), color)
                color = (255 * (nbpts - i - 1) / nbpts, 0, 255 * (i + 1) / nbpts)

    def _check_hull(self):
        """ If self._points form a convex hull of 4 vertices, store these into self.hull. Otherwise erase self.hull.
        """
        if 3 < len(self._points):
            self.hull = imgutil.get_ordered_hull(self._points)
            if len(self.hull) != 4:
                self.hull = None
        else:
            self.hull = None

    def __str__(self):
        return "Corners:" + str(self._points)
