from queue import Empty

import cv2
from numpy import zeros_like, zeros, int32, empty
from time import time

from camkifu.core.imgutil import draw_str
from golib.model.move import Move
from camkifu.stone.stonesfinder import StonesFinder, compare, evalz
from golib.config.golib_conf import gsize, B, W, E


__author__ = 'Arnaud Peloquin'

# the minimum mean color difference to trigger motion detection when in watching state
watch_diff_threshold = 90
# the minimum mean color difference to trigger motion detection when in searching state
search_diff_threshold = 100

# possible states for a BackgroundSub instance :
sampling = "sampling"
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
        self.lastpos = None

        self.state = sampling
        self.nb_untouched = 0  # the number of successive searches that detected no motion at all
        self.last_on = time()  # instant when last active. to be used to detect long sleeps.

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)
        filtered *= self.getmask(filtered)
        if self.state == sampling:
            done = self.sample(filtered)
            if done:
                self.state = searching
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
        draw_str(filtered, 40, 60, "state : " + self.state)
        self._show(filtered)

    def _learn(self):
        try:
            while True:
                # todo implement correction in case of deletion by user (see base method doc).
                err, exp = self.corrections.get_nowait()
                print("%s has become %s" % (err, exp))
        except Empty:
            pass

    def reset(self):
        """
        Clear the cached background and the last positive position.

        """
        self._background = zeros_like(self._background)
        self.lastpos = None
        self.state = sampling

    def sample(self, img):
        """
        Return True when finished updating the background data (mean color) of each empty intersection.
        Occupied intersections are left untouched -> better to have old background data than stone data.
        The method should be called until it returns true (may need several passes)

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
        return True

    def compare_pos(self, img, x, y):
        """
        Return an evaluation of the mean difference between img and the background for the zone around (x,y).
        Note : the use of the mean implies that the difference is normalized by the zone surface.

        img   -- the image to evaluate.
        x, y  -- the indexes of the zone (i.e. the coordinate of the goban intersection).

        """
        zone, points = self._getzone(img, x, y)
        # noinspection PyNoneFunctionAssignment
        mean_eval = empty((3), dtype=self._background.dtype)
        for chan in range(3):
            mean_eval[chan] = evalz(zone, chan) / self.zone_area
        delta = compare(self._background[x][y], mean_eval)
        return delta

    def watch(self, img):
        """
        Idle version of the stone detection that only checks the border.

        The idea is that after several passes of full search detecting no motion, performance can be improved
        by checking the first (outer) line only : a hand cannot reach inner lines without disturbing the first line.

        """
        assert self.state == watching, "State allowed to enter method : watching"
        for x, y in self._empties_border(0):
            delta = self.compare_pos(img, x, y)
            if not -watch_diff_threshold < delta < watch_diff_threshold:
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
        # deltas = []
        # todo analyse one intersection out of two in order to speed up search, and have a method to search around a
            # given targeted intersection to detect disturbance motion faster. See if one row is enough for that.
        for x, y in self.empties_spiral():
            delta = self.compare_pos(img, x, y)

            if not -search_diff_threshold < delta < search_diff_threshold:
                self.nb_untouched = 0
                color = B if delta < 0 else W
                if pos is None:
                    pos = x, y
                else:
                    # todo allow for 1 (maybe 2) neighbour stones to have been polluted by current stone, and ignore ??
                    # or better along the same idea : when the area is spotted, try to move the coordinates around to
                    # see if there is one area standing out. because sometimes the grid is not well perfectly and a
                    # stone overlaps two positions.
                    print("dropped frame: {0} (2 hits)".format(self.__class__.__name__))
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
                self.state = sampling
            else:
                self.lastpos = pos
        else:
            self.nb_untouched += 1

    def _window_name(self):
        return BackgroundSub.label


def sampled(img):
    print("Image at {0} set as background.".format(hex(id(img))))