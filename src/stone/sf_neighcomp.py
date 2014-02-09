from Queue import Empty
from collections import defaultdict
import cv2
from numpy import int32, zeros
from go.move import Move
from stone.stonesfinder import StonesFinder, evalz
from golib_conf import gsize, E

__author__ = 'Arnaud Peloquin'


class NeighbourComp(StonesFinder):
    """
    A few tries at determining the state (B W or E) of an intersection by comparing it against its neighbours.

    Not finished.

    """

    label = "Neigh Comp"

    def __init__(self, vmanager):
        super(NeighbourComp, self).__init__(vmanager)
        self.lastpos = None

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)
        # filtered = goban_img
        filtered *= self.getmask(filtered)
        disp_img = filtered.copy()

        # evaluate each empty intersection of the goban
        sample = zeros((gsize, gsize, 3), int32)
        empties = list(self.empties())  # todo is caching still needed ?

        for x in range(gsize):
            for y in range(gsize):
                zone, _ = self._getzone(filtered, x, y)
                for chan in range(3):
                    sample[x][y][chan] = evalz(zone, chan)
        sample /= self.zone_area  # use mean

        pos = None
        color = None
        # deltas = []
        # values = zeros((gsize, gsize), dtype=int32)
        colors = self.getcolors()
        for x, y in empties:
            neighs = defaultdict(default_factory=lambda: [])
            for k, l in neighbours(x, y):
                neighs[colors[k][l]] = sample[k, l]
            current_color = self.compute_color(neighs)
            if current_color != E:
                if pos is None:
                    color = current_color
                    pos = x, y
                else:
                    print "dropped frame: {0} (2 hits)".format(self.__class__.__name__)
                    # self._drawvalues(disp_img, sample)
                    # self._show(disp_img, name="Goban frame")
                    pos = None
                    break

            # insort(deltas, delta)
            # values[x, y] = delta / 10
        # length = len(deltas)
        # print str(deltas[0:5]) + str(deltas[length - 5:length])

        if pos is not None:
            if self.lastpos == pos:
                self.suggest(Move("cv", ctuple=(color, pos[0], pos[1])))
            else:
                self.lastpos = pos

        # self._drawvalues(disp_img, values)
        self._show(disp_img, name="Goban frame")

    def _learn(self):
        try:
            while True:
                err, exp = self.corrections.get_nowait()
                print "%s has become %s" % (err, exp)
        except Empty:
            pass

    @staticmethod
    def compute_color(neighs):
        pass


def neighbours(x, y):
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i or j != 0:
                if (0 <= x + i < gsize) and (0 <= y + j < gsize):
                    yield x + i, y + j