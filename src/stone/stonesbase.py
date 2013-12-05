import cv2
from numpy import zeros_like, zeros, uint8, int16, sum as npsum
from numpy.ma import absolute
from board.boardbase import ordered_hull
from config.devconf import canonical_size
from core.imgutil import draw_circles
from core.video import VidProcessor
from golib_conf import gsize

__author__ = 'Kohistan'


class StonesFinder(VidProcessor):
    """
    Abstract class providing a structure for stones-finding processes.

    """

    def __init__(self, vmanager, rect):
        super(StonesFinder, self).__init__(vmanager, rect)

        self.stones = zeros((gsize, gsize), dtype=uint8)
        self._grid = Grid(canonical_size)

    def _doframe(self, frame):
        transform = self.vmanager.board_finder.mtx
        if transform is not None:
            goban_img = cv2.warpPerspective(frame, transform, (canonical_size, canonical_size))
            self._find(goban_img)

    def _find(self, img):
        raise NotImplementedError("Abstract method meant to be extended")

    def reset(self):
        self.stones = zeros_like(self.stones)

    def _getzones(self, img, r, c):
        """
        Returns the pixel zone corresponding to the given goban intersection.
        The current approximation of a stone area is a cross (optimally should be a disk)

        img -- expected to contain the goban pixels only, in the canonical frame.
        r -- the intersection row index, numpy-like
        c -- the intersection column index, numpy-like

        """
        d, u = 1, 1  # the weights to give to vertical directions (down, up)
        proportions = [0.5, 1.0]  # needa be floats

        p = self._grid.pos[r][c]
        pbefore = self._grid.pos[r - 1][c - 1].copy()
        pafter = self._grid.pos[min(r + 1, gsize - 1)][min(c + 1, gsize - 1)].copy()
        if r == 0:
            pbefore[0] = -p[0]
        elif r == gsize - 1:
            pafter[0] = 2 * img.shape[0] - p[0] - 2
        if c == 0:
            pbefore[1] = -p[1]
        elif c == gsize - 1:
            pafter[1] = 2 * img.shape[1] - p[1] - 2

        rects = []
        points = []  # for debugging purposes. may be a good thing to clean that out
        # determine start and end point of the rectangle
        for i in range(2):
            w1 = proportions[i] / 2
            w2 = proportions[1-i] / 2
            start = (int(w1*pbefore[0] + (1-w1)*p[0]), int(w2*pbefore[1] + (1-w2)*p[1]))
            end = (int((1-w1)*p[0] + w1*pafter[0]), int((1-w2)*p[1] + w2*pafter[1]))
            rects.append(img[start[0]: end[0], start[1]: end[1]].copy())  # todo try to remove this copy()
            points.append((start[1], start[0], end[1], end[0]))

        return rects, points

    def _drawgrid(self, img):
        if self._grid is not None:
            centers = []
            for i in range(19):
                for j in range(19):
                    centers.append(self._grid.pos[i][j])
                    draw_circles(img, centers)


class Grid(object):
    """
    Store the location of each intersection of the goban.
    The aim of splitting that part in a separate class is to allow for a more automated and robust
    version to be developed.

    """

    def __init__(self, size):
        self.pos = zeros((gsize, gsize, 2), dtype=int16)
        # the 2 lines below would benefit from some sort of automation
        start = size / gsize / 2
        end = size - start

        hull = ordered_hull([(start, start), (end, start), (end, end), (start, end)])
        assert len(hull) == 4, "The points expected here are the 4 corners of the grid."
        for i in range(gsize):
            xup = (hull[0][0] * (gsize - 1 - i) + hull[1][0] * i) / (gsize - 1)
            xdown = (hull[3][0] * (gsize - 1 - i) + hull[2][0] * i) / (gsize - 1)
            for j in range(gsize):
                self.pos[i][j][0] = (xup * (gsize - 1 - j) + xdown * j) / (gsize - 1)

                yleft = (hull[0][1] * (gsize - 1 - j) + hull[3][1] * j) / (gsize - 1)
                yright = (hull[1][1] * (gsize - 1 - j) + hull[2][1] * j) / (gsize - 1)
                self.pos[i][j][1] = (yleft * (gsize - 1 - i) + yright * i) / (gsize - 1)


def compare(reference, current):
    """
    Return a distance between the two colors. The value is positive if current is
    brighter than the reference, and negative otherwise.

    background -- a vector of length 3
    current -- a vector of length 3

    """
    sign = 1 if npsum(reference) <= npsum(current) else -1
    return sign * int(npsum(absolute(current - reference)))