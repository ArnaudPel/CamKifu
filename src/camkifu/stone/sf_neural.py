
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
        stones = self.manager.predict_stones(goban_img)
        moves = []
        for i in range(gsize):
            for j in range(gsize):
                moves.append((stones[i][j], i, j))
        self.bulk_update(moves)
