import numpy as np

from camkifu.stone import StonesFinder
from camkifu.stone.tmanager import TManager
from golib.config.golib_conf import gsize

__author__ = 'Arnaud Peloquin'


class SfNeural(StonesFinder):

    def __init__(self, vmanager, learn_bg=True):
        super().__init__(vmanager, learn_bg=learn_bg)
        self.manager = TManager()
        self.frame_period = 1  # todo use more sophisticated background check as trigger

    def _find(self, goban_img):
        x_s = self.manager.generate_xs(goban_img)
        predict = self.manager.predict(x_s)
        stones = np.ndarray((gsize, gsize), dtype=object)
        for k, arr in enumerate(predict):
            i = int(k / self.manager.split)
            j = k % self.manager.split
            rs, re, cs, ce = self.manager._subregion(i, j)
            stones[rs:re, cs:ce] = arr.reshape((2, 2))
        moves = []
        for i in range(gsize):
            for j in range(gsize):
                moves.append((stones[i][j], i, j))
        self.bulk_update(moves)
