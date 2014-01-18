from Queue import Empty
from bisect import insort
import cv2
from numpy import zeros_like, zeros, uint8, int32, empty, empty_like, mean, sum as npsum
from numpy.ma import absolute
from go.move import Move
from stone.stonesfinder import StonesFinder, compare, evalz
from golib_conf import gsize, B, W, E

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
            # self._drawgrid(filtered)
        self._show(filtered, name="Goban frame")

    def _learn(self):
        try:
            while True:
                err, exp = self.corrections.get_nowait()
                print "%s has become %s" % (err, exp)
        except Empty:
            pass

    def reset(self):
        self._background = zeros_like(self._background)
        self.lastpos = None
        self.dosample = True

    def sample(self, img):
        """
        Update the background data (mean color) of each empty intersection.
        Occupied intersections are left untouched -> better to have old background data than setting a stone as bg.

        """
        for x, y in self.empties():
            zone, points = self._getzone(img, x, y)
            #copy = img.copy()
            for chan in range(3):
                self._background[x, y, chan] = evalz(zone, chan) / self.zone_area
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
        # todo improvement: only check for one (maybe 2) lines around goban. As long as they are
        #   undisturbed, it means a stone cannot possibly have been put.
        #   Unless thread has been interrupted for very long though
        assert len(self._background) == gsize, "At least one sample must have been run to provide comparison data."
        pos = None
        color = E
        # subtract = zeros_like(img)
        sample = empty_like(self._background)
        # deltas = []
        for x, y in self.empties():
            zone, points = self._getzone(img, x, y)
            for chan in range(3):
                sample[x, y, chan] = evalz(zone, chan) / self.zone_area
            delta = compare(self._background[x][y], sample[x, y])

            if not -100 < delta < 100:
                color = B if delta < 0 else W
                if pos is None:
                    pos = x, y
                else:
                    # todo allow for 1 (maybe 2) neighbour stones to have been polluted by current stone, and ignore ??
                    print "dropped frame: {0} (2 hits)".format(self.__class__.__name__)
                    return

                    # insort(deltas, delta)
            # length = len(deltas)
        # print str(deltas[0:5]) + str(deltas[length-5:length])

        # subtract = zeros_like(img)
        #     current -= bg
        #     current = absolute(current)
        #     color = (int(current[0]), int(current[1]), int(current[2]))
        #     cv2.rectangle(subtract, points[0:2], points[2:4], color, thickness=-1)
        # self._show(subtract, name="Subtract")

        if pos is not None:
            if self.lastpos == pos:
                self.suggest(Move("cv", ctuple=(color, pos[0], pos[1])))
                sample[pos[0], pos[1]] = self._background[pos[0], pos[1]]  # don't sample stone found as background
                self._background = sample
                sampled(img)
            else:
                self.lastpos = pos


def sampled(img):
    print "Image at {0} set as background.".format(hex(id(img)))






















