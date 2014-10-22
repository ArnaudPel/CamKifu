from Queue import Queue, Full

import cv2
from numpy import zeros, uint8, int16, sum as npsum, empty, ogrid
from numpy.ma import absolute, empty_like

from camkifu.config.cvconf import canonical_size
from camkifu.core.imgutil import draw_circles, draw_str, order_hull
from camkifu.core.video import VidProcessor
from golib.config.golib_conf import gsize, E


__author__ = 'Arnaud Peloquin'


correc_size = 10


class StonesFinder(VidProcessor):
    """
    Abstract class providing a base structure for stones-finding processes.

    """

    def __init__(self, vmanager):
        super(StonesFinder, self).__init__(vmanager)
        self._posgrid = PosGrid(canonical_size)
        self.mask_cache = None
        self.zone_area = None  # the area of a zone # (non-zero pixels of the mask)
        self.corrections = Queue(correc_size)

    def _doframe(self, frame):
        transform = None
        if self.vmanager.board_finder is not None:
            transform = self.vmanager.board_finder.mtx
        if transform is not None:
            goban_img = cv2.warpPerspective(frame, transform, (canonical_size, canonical_size))
            self._learn()
            self._find(goban_img)
        # todo implement 'else' to indicate that there is no board location available

    def _find(self, goban_img):
        """
        Detect stones in the (already) canonical image of the goban.

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _learn(self):
        """
        Process corrections queue, and perform algorithm adjustments if necessary.

        This choice to "force" implementation using an abstract method is based on the fact that stones
        deleted by the user MUST be acknowledged and dealt with, in order not to be re-suggested straight away.
        Added stones are not so important, because their presence is automatically reflected in self.empties().

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def suggest(self, move):
        """
        Suggest the add of a new stone to the goban.

        """
        print move
        self.vmanager.controller.pipe("append", [move])

    def corrected(self, err_move, exp_move):
        """
        Entry point to provide corrections made by the user to the stones on the Goban. See _learn().

        """
        try:
            self.corrections.put_nowait((err_move, exp_move))
        except Full:
            print "Corrections queue full (%s), ignoring %s -> %s" % (correc_size, str(err_move), str(exp_move))

    def empties(self):
        """
        Yields the unoccupied positions of the goban in naive order.
        Note: this implementation allows for the positions to be updated by another thread during yielding.

        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.rules[x][y] == E:
                    yield y, x

    def empties_spiral(self):
        """
        Yields the unoccupied positions of the goban along an inward spiral.
        Aims to help detect hand / arm appearance faster by analysing outer border(s) first.

        """
        inset = 0
        while inset <= gsize / 2:
            for x, y in self._empties_border(inset):
                yield x, y
            inset += 1

    def _empties_border(self, inset):
        """
        Yields the unoccupied positions of the goban along an inward spiral.

        inset -- the inner margin defining the start of the inward spiral [0 = outer border -> gsize/2 = center position].
                 it always ends at the center of the goban.

        """
        y = inset
        for x in range(inset, gsize - inset):
            # todo extract "do_yield()" method to remove code duplicate ?
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        x = gsize - inset - 1
        for y in range(inset + 1, gsize - inset):
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        y = gsize - inset - 1
        for x in range(gsize - inset - 2, inset - 1, -1):  # reverse just to have a nice spiral. not actually useful
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        x = inset
        for y in range(gsize - inset - 2, inset, -1):
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

    def getcolors(self):
        """
        Return a copy of the current goban state.

        """
        return self.vmanager.controller.rules.copystones()

    def _getzone(self, img, r, c, cursor=1.0):
        """
        Returns the (rectangle) pixel zone corresponding to the given goban intersection.

        img -- expected to contain the goban pixels only, in the canonical frame.
        r -- the intersection row index
        c -- the intersection column index
        cursor -- must be float, has sense in the interval ]0, 2[
                  0 -> the zone is restricted to the (r, c) point.
                  2 -> the zone is delimited by the rectangle (r-1, c-1), (r+1, c+1).
                  1 -> the zone is a rectangle of "intuitive" size, halfway between the '0' and '2' cases.

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

    def getmask(self, frame):
        """
        A boolean mask the size of "frame" that has a circle around each goban intersection.
        Multiply a frame by this mask to zero-out anything outside the circles.

        """
        if self.mask_cache is None:
            # todo : observation shows that stones of the front line are seen too high (due to cam angle most likely)
            # investigate more and see to adapt the repartition of the mask ? Some sort of vertical gradient of size or
            # location. The former will imply the introduction of a structure to store all zones areas, at least one
            #  per line.
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

    def _drawgrid(self, img):
        """
        Draw a circle around each intersection of the goban, as they are currently estimated.

        """
        if self._posgrid is not None:
            centers = []
            for i in range(19):
                for j in range(19):
                    centers.append(self._posgrid[i][j])
            draw_circles(img, centers)

    def _drawvalues(self, img, values):
        """
        Display one value per goban position. Obviously values will soon overlap if they are longish.

        """
        for row in range(gsize):
            for col in range(gsize):
                x, y = self._posgrid[row, col]
                draw_str(img, (x - 10, y + 2), str(values[row, col]))


def evalz(zone, chan):
    """ Return an integer evaluation for the zone. """
    return int(npsum(zone[:, :, chan]))


def compare(reference, current):
    """
    Return a distance between the two colors. The value is positive if current is
    brighter than the reference, and negative otherwise.

    reference -- a vector
    current -- a vector of same length as reference.

    """
    sign = 1 if npsum(reference) <= npsum(current) else -1
    return sign * int(npsum(absolute(current - reference)))


class PosGrid(object):
    """
    Store the location of each intersection of the goban.
    Can be extended to provide an evolutive version that can learn on the flight.

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