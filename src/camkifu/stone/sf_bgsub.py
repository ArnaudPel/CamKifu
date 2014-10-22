from Queue import Empty

import cv2
from numpy import zeros_like, zeros, int32, empty_like
from time import time

from camkifu.core.imgutil import draw_str
from golib.model.move import Move
from camkifu.stone.stonesfinder import StonesFinder, compare, evalz
from golib.config.golib_conf import gsize, B, W, E


__author__ = 'Arnaud Peloquin'


# possible states for a BackgroundSub instance :
watching = "watching"
searching = "searching"

# max number of 'untouched' searches before triggering the 'watching' state
untouched_threshold = 7


class BackgroundSub(StonesFinder):
    """
    Save background data using sample(img).
    Perform background subtraction operations in order to detect stones.

    """

    label = "Bg Sub"

    def __init__(self, vmanager):
        super(BackgroundSub, self).__init__(vmanager)
        self.bindings['s'] = self.reset

        self._background = zeros((gsize, gsize, 3), dtype=int32)
        self.dosample = True
        self.lastpos = None

        self.zone_area = None  # the area of a zone # (non-zero pixels of the mask)
        self.state = searching
        self.nb_untouched = 0  # the number of successive searches that detected no motion at all
        self.last_on = time()  # instant when last active. to be used to detect long sleeps.

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)
        filtered *= self.getmask(filtered)
        if self.dosample:
            self.sample(filtered)
            self.dosample = False
        else:
            # force full search if the last processing is too long ago
            if 2 < time() - self.last_on:
                self.state = searching
            if self.state == searching:
                self.search(filtered)
                if untouched_threshold <= self.nb_untouched:
                    self.nb_untouched = 0
                    self.state = watching
            else:
                self.watch(filtered)
            # self._drawgrid(filtered)
        self.last_on = time()
        draw_str(filtered, (40, 60), "state : " + self.state)
        self._show(filtered, name="Goban frame")

    def _learn(self):
        try:
            while True:
                err, exp = self.corrections.get_nowait()
                print "%s has become %s" % (err, exp)
        except Empty:
            pass

    def reset(self):
        """
        Clear the cached background and the last positive position.

        """
        self._background = zeros_like(self._background)
        self.lastpos = None
        self.dosample = True

    def sample(self, img):
        """
        Update the background data (mean color) of each empty intersection.
        Occupied intersections are left untouched -> better to have old background data than stone data.

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

    def compare_pos(self, evals, img, x, y):
        """
        Return an evaluation of the difference between img and the background for the zone (x,y).
        evals -- a matrix that will be updated at position (x,y) with the evaluation of that zone.
        img -- the image to evaluate.
        x, y -- the indexes of the zone (i.e. the coordinate of the goban intersection).

        """
        zone, points = self._getzone(img, x, y)
        for chan in range(3):
            evals[x, y, chan] = evalz(zone, chan) / self.zone_area
        delta = compare(self._background[x][y], evals[x, y])
        return delta

    def watch(self, img):
        """
        Idle version of the stone detection that only checks the border.

        The idea is that after several passes of full search detecting no motion, performance can be improved
        by checking the first (outer) line only : a hand cannot reach inner lines without disturbing the first line.

        """
        # todo check for thread sleep (if too long, swap to 'searching' state)
        assert self.state == watching, "State allowed to enter method : watching"
        sample = empty_like(self._background)
        for x, y in self._empties_border(0):
            delta = self.compare_pos(sample, img, x, y)
            if not -70 < delta < 70:
                self.state = searching
                break

    def search(self, img):
        """
        Try to detect stones by comparing against (cached) background colors.

        """
        assert len(self._background) == gsize, "At least one sample must have been run to provide comparison data."
        pos = None
        color = E
        # subtract = zeros_like(img)  # debug variable, see below for usage.
        sample = empty_like(self._background)
        # deltas = []
        for x, y in self.empties_spiral():
            delta = self.compare_pos(sample, img, x, y)

            # todo make thresholds relative to something ??
            if not -100 < delta < 100:
                self.nb_untouched = 0
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
        else:
            self.nb_untouched += 1


def sampled(img):
    print "Image at {0} set as background.".format(hex(id(img)))