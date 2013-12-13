from bisect import insort
import cv2
from numpy import zeros_like, zeros, uint8, int32, empty, empty_like, mean, sum as npsum
from numpy.ma import absolute
from stone.stonesbase import StonesFinder, compare, evalz
from golib_conf import gsize, player_color

__author__ = 'Kohistan'


class BackgroundSub(StonesFinder):
    """
    Save background data using sample(img)
    Perform background subtraction operations in order to detect stones.

    self.stones -- the matrix of stones found so far (0:None, 1:Black, 2:White)
    """

    def __init__(self, vmanager, rectifier):
        super(BackgroundSub, self).__init__(vmanager, rectifier)
        self.bindings['s'] = self.reset

        self._background = zeros((gsize, gsize, 3), dtype=int32)
        self.dosample = True
        self.lastpos = None

        self.zone_area = None  # the area of a zone # (non-zero pixels of the mask)

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)
        filtered *= self.getmask(filtered)
        if self.dosample:
            self.sample(filtered)
            self.dosample = False
        else:
            self.detect(filtered)
        self._drawgrid(filtered)
        self._show(filtered, name="Goban frame")

    def reset(self):
        self._background = zeros_like(self._background)
        self.lastpos = None
        self.dosample = True

    def sample(self, img):
        for x in range(gsize):
            for y in range(gsize):
                zone, points = self._getzones(img, x, y)
                #copy = img.copy()
                for chan in range(3):
                    self._background[x, y, chan] = evalz(chan, zone)
        self._background /= self.zone_area
                #cv2.rectangle(copy, points[0:2], points[2:4], (255, 0, 0), thickness=-1)
                #self._show(copy, name="Sampling Zone")
                #if cv2.waitKey() == 113: raise SystemExit()
        sampled(img)

    def detect(self, img):
        """
        Try to detect stones by comparing against neighbour colors.

        mask -- a matrix of shape (gsize, gsize) providing positions of already known stones, as follow.
                0: empty position
                1: white position
                2: black position

        """
        assert len(self._background) == gsize, "At least one sample must have been run to provide comparison data."
        pos = None
        val = 0
        # subtract = zeros_like(img)
        sample = empty_like(self._background)
        deltas = []
        for x, y in self.empties():
            zone, points = self._getzones(img, x, y)
            for chan in range(3):
                sample[x, y, chan] = evalz(chan, zone) / self.zone_area  # needs to be divided now
            delta = compare(self._background[x][y], sample[x, y])

            if delta < -100 or 100 < delta:
                val = 1 if delta < 0 else 2
                if pos is None:
                    pos = x, y
                else:
                    # todo allow for one (maybe two) neighbour stones to have been polluted by current stone, and ignore ??
                    print "dropped frame: StonesFinder (2 hits)"
                    return

            insort(deltas, delta)
        length = len(deltas)
        print str(deltas[0:5]) + str(deltas[length-5:length])

        # subtract = zeros_like(img)
        #     current -= bg
        #     current = absolute(current)
        #     color = (int(current[0]), int(current[1]), int(current[2]))
        #     cv2.rectangle(subtract, points[0:2], points[2:4], color, thickness=-1)
        # self._show(subtract, name="Subtract")

        if pos is not None:
            if self.lastpos == pos:
                self.suggest(player_color[val], pos[1], pos[0])  # purposely rotated
                sample[pos[0], pos[1]] = self._background[pos[0], pos[1]]  # don't sample stone found as background
                self._background = sample
                sampled(img)
            else:
                self.lastpos = pos


def sampled(img):
    print "Image at {0} set as background.".format(hex(id(img)))






















