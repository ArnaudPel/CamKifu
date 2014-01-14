from threading import RLock
from time import time, sleep
import cv2
from numpy import zeros, uint8, int16, sum as npsum, zeros_like, empty, ogrid, ones, sum
from numpy.ma import absolute, empty_like
from board.boardbase import order_hull
from config.devconf import canonical_size
from core.imgutil import draw_circles, draw_str
from core.video import VidProcessor
from go.sgf import Move
from golib_conf import gsize

__author__ = 'Kohistan'


class StonesFinder(VidProcessor):
    """
    Abstract class providing a structure for stones-finding processes.

    """

    def __init__(self, vmanager, rect):
        super(StonesFinder, self).__init__(vmanager, rect)
        self._posgrid = PosGrid(canonical_size)
        self.mask_cache = None
        self.zone_area = None

    def _doframe(self, frame):
        transform = self.vmanager.board_finder.mtx
        if transform is not None:
            goban_img = cv2.warpPerspective(frame, transform, (canonical_size, canonical_size))
            self._find(goban_img)

    def _find(self, goban_img):
        raise NotImplementedError("Abstract method meant to be extended")

    def _drawgrid(self, img):
        if self._posgrid is not None:
            centers = []
            for i in range(19):
                for j in range(19):
                    centers.append(self._posgrid[i][j])
            draw_circles(img, centers)

    def _drawvalues(self, img, values):
        for row in range(gsize):
            for col in range(gsize):
                x, y = self._posgrid[row, col]
                draw_str(img, (x - 10, y + 2), str(values[row, col]))

    def suggest(self, move):
        """
        Suggest the add of a new stone to the goban.

        """
        print move
        self.vmanager.controller.pipe("append", [move])

    def empties(self):
        """
        Enumerate the unoccupied positions of the goban.
        Note: May be updated by another thread while yielding results.

        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.rules[x][y] == 'E':
                    yield y, x

    def getcolors(self):
        return self.vmanager.controller.rules.copystones()

    def _getzone(self, img, r, c, cursor=1.0):
        """
        Returns the pixel zone corresponding to the given goban intersection.
        The current approximation of a stone area is a cross (optimally should be a disk)

        img -- expected to contain the goban pixels only, in the canonical frame.
        r -- the intersection row index, numpy-like
        c -- the intersection column index, numpy-like
        proportions -- must be floats

        """
        assert isinstance(cursor, float)
        p = self._posgrid[r][c]
        pbefore = self._posgrid[r - 1][c - 1].copy()
        pafter = self._posgrid[min(r + 1, gsize - 1)][min(c + 1, gsize - 1)].copy()
        if r == 0:
            pbefore[0] = -p[0]
        elif r == gsize - 1:
            pafter[0] = 2 * img.shape[0] - p[0] - 2
        if c == 0:
            pbefore[1] = -p[1]
        elif c == gsize - 1:
            pafter[1] = 2 * img.shape[1] - p[1] - 2

        # determine start and end point of the rectangle
        w = cursor / 2
        sx = max(0, int(w * pbefore[0] + (1 - w) * p[0]))
        sy = max(0, int(w * pbefore[1] + (1 - w) * p[1]))
        ex = min(img.shape[0], int((1 - w) * p[0] + w * pafter[0]))
        ey = min(img.shape[1], int((1 - w) * p[1] + w * pafter[1]))

        # todo remove this copy() and leave it to caller
        return img[sx: ex, sy: ey].copy(), (sx, sy, ex, ey)

    # def _getmask(self, img, r, c):
    #     mask = empty_like(img, dtype=bool)
    #     a, b = self._posgrid[r][c]
    #
    #     r2 = r + 1 if r < gsize - 1 else r - 1
    #     c2 = c + 1 if c < gsize - 1 else c - 1
    #     a1, b1 = self._posgrid[r2][c2]
    #     rad = min(abs(a-a1), abs(b-b1)) / 2
    #
    #     y, x = ogrid[-a:img.shape[0]-a, -b: img.shape[1] - b]  # todo check if there isn't a more direct way
    #     layer = x*x + y*y <= rad*rad
    #
    #     nblayers = img.shape[2] if 2 < len(img.shape) else 1
    #     for z in range(nblayers):
    #         mask[:, :, z] = layer
    #     return mask
    #
    def getmask(self, frame):
        if self.mask_cache is None:
            print "initializing mask"
            self.mask_cache = empty_like(frame)
            mask = empty(frame.shape[0:2], dtype=uint8)
            for row in range(gsize):
                for col in range(gsize):
                    zone, (sx, sy, ex, ey) = self._getzone(frame, row, col)  # todo expose proportions ?
                    a = zone.shape[0] / 2
                    b = zone.shape[1] / 2
                    r = min(a, b)
                    y, x = ogrid[-a:zone.shape[0] - a, -b: zone.shape[1] - b]
                    zmask = x * x + y * y <= r * r
                    mask[sx: ex, sy: ey] = zmask

            # duplicate mask to match image depth
            for i in range(self.mask_cache.shape[2]):
                self.mask_cache[:, :, i] = mask

            # store the area of one zone for normalizing purposes
            zone, _ = self._getzone(mask, 0, 0)
            self.zone_area = npsum(zone)
            print "area={0}".format(self.zone_area)

        return self.mask_cache


def evalz(zone, chan):
    return int(npsum(zone[:, :, chan]))


class PosGrid(object):
    """
    Store the location of each intersection of the goban.
    Can be extended to provide an evolutive version.

    """

    def __init__(self, size):
        self.pos = zeros((gsize, gsize, 2), dtype=int16)
        # the 2 lines below would benefit from some sort of automation
        start = size / gsize / 2
        end = size - start

        hull = order_hull([(start, start), (end, start), (end, end), (start, end)])
        assert len(hull) == 4, "The points expected here are the 4 corners of the grid."
        for i in range(gsize):
            xup = (hull[0][0] * (gsize - 1 - i) + hull[1][0] * i) / (gsize - 1)
            xdown = (hull[3][0] * (gsize - 1 - i) + hull[2][0] * i) / (gsize - 1)
            for j in range(gsize):
                self[i][j][0] = (xup * (gsize - 1 - j) + xdown * j) / (gsize - 1)

                yleft = (hull[0][1] * (gsize - 1 - j) + hull[3][1] * j) / (gsize - 1)
                yright = (hull[1][1] * (gsize - 1 - j) + hull[2][1] * j) / (gsize - 1)
                self[i][j][1] = (yleft * (gsize - 1 - i) + yright * i) / (gsize - 1)

    def __getitem__(self, item):
        return self.pos.__getitem__(item)

    def __getslice__(self, i, j):
        return self.pos.__getslice__(i, j)


class ScoreGrid(object):
    # todo park that somewhere else or del
    """
    Can be used to arbitrate between several stone detection algorithms.
    Values are automatically deprecated, based on their age.

    """

    def __init__(self):
        self._grids = {}
        self._thresholds = {}
        self._rlock = RLock()

        self.deprec_time = 3.0  # number of seconds a value is regarded as meaningful

    def get_all(self):
        totals = zeros((gsize, gsize), dtype=int16)  # the grid of total score for each stone
        with self._rlock:
            thresh = sum(self._thresholds.itervalues())
            tps = time()
            for grid in self._grids.values():
                totals += (grid[::0] * self.deprec_time) / (-grid[::1] + tps)
            totals[totals < thresh] = 0  # mask values under threshold
            return totals

    def set(self, key, grid, thresh_contrib):
        with self._rlock:  # may not be needed here, but just in case
            self._grids[key] = grid
            self._thresholds[key] = thresh_contrib

    def delete(self, key):
        del self._grids[key]
        del self._thresholds[key]


def compare(reference, current):
    """
    Return a distance between the two colors. The value is positive if current is
    brighter than the reference, and negative otherwise.

    background -- a vector of length 3
    current -- a vector of length 3

    """
    sign = 1 if npsum(reference) <= npsum(current) else -1
    return sign * int(npsum(absolute(current - reference)))


class DummyFinder(StonesFinder):
    """
    Can be used to simulate the detection of an arbitrary sequence of stones.
    Useful to test "test code". Double use of word 'test' intended :)

    """

    def __init__(self, vmanager, rect, ctype, sequence):
        super(DummyFinder, self).__init__(vmanager, rect)
        self.iterator = iter(sequence)
        self.ctype = ctype

    def _find(self, goban_img):
        try:
            move = self.iterator.next()
            sleep(2)  # wait for a (potential) gui to be initialized
            self.suggest(Move(self.ctype, string=move))
        except StopIteration:
            self.interrupt()
