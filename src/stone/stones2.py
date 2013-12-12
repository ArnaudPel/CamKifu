from bisect import insort
import numpy as np
from stone.stonesbase import StonesFinder, compare
from golib_conf import gsize

__author__ = 'Kohistan'


class NeighbourComp(StonesFinder):
    def __init__(self, vmanager, rect):
        super(NeighbourComp, self).__init__(vmanager, rect)
        self._colors = np.zeros((gsize, gsize, 3), np.int16)

    def _find(self, img):
        self._drawgrid(img)
        self._show(img)
        # record the mean color for each intersection of the goban.
        for x in range(gsize):
            for y in range(gsize):
                zone, _ = self._getzones(img, x, y)
                for chan in range(3):
                    self._colors[x][y][chan] = int(np.mean(zone[:, :, chan]))

        deltas = []
        for x, y in self.empties():
            #r = np.zeros(8, np.int16)
            #g = np.zeros(8, np.int16)
            #b = np.zeros(8, np.int16)
            r = []
            g = []
            b = []
            idx = 0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if i or j != 0:
                        try:
                            meancol = self._colors[x + i][y + j]
                            #r[idx] = meancol[0]
                            #g[idx] = meancol[1]
                            #b[idx] = meancol[2]
                            r.append(meancol[0])
                            g.append(meancol[1])
                            b.append(meancol[2])
                        except IndexError:
                            pass

                        idx += 1
            neighcol = (np.mean(r), np.mean(g), np.mean(b))
            insort(deltas, compare(neighcol, self._colors[x][y]))
        length = len(deltas)
        print str(deltas[0:5]) + str(deltas[length - 5:length])





