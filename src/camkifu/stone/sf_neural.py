import cv2
import numpy as np

from camkifu.stone import StonesFinder
from camkifu.stone.nn_manager import NNManager
from golib.config.golib_conf import gsize

__author__ = 'Arnaud Peloquin'


class SfNeural(StonesFinder):

    def __init__(self, vmanager):
        super().__init__(vmanager, learn_bg=True)
        self.manager = NNManager()
        self.frame_period = 1  # todo use more sophisticated background check as trigger
        self.candidates = np.zeros((gsize, gsize), dtype=np.uint8)

    def _find(self, goban_img):
        # if self.total_f_processed == 1:  # todo is this value updated during live ?
        #     self.predict_all(goban_img)
        if self.total_f_processed < self.bg_init_frames:
            self.display_bg_sampling(goban_img.shape)
        else:
            canvas = goban_img.copy()
            r, w, _ = goban_img.shape
            if r * w * 0.05 < np.sum(self.get_foreground()) / 255:
                self.wait(canvas)  # wait for things to settle
            else:
                self.predict(canvas=canvas)
            self._show(canvas, name='Neural')

    def predict_all(self, goban_img):
        stones = self.manager.predict_stones(goban_img)
        moves = []
        for i in range(gsize):
            for j in range(gsize):
                moves.append((stones[i][j], i, j))
        self.bulk_update(moves)

    def predict(self, canvas: np.ndarray = None, color=(0, 255, 0)):
        fg = self.get_foreground()
        for x in range(gsize):
            for y in range(gsize):
                a0, b0, a1, b1 = self.getrect(x, y)
                if (a1 - a0) * (b1 - b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255:
                    self.candidates[x, y] += 1
                    c = color
                    if 2 < self.candidates[x, y]:
                        c = (0, 255, 255)
                        self.candidates[x, y] = 0
                    if canvas is not None:
                        cv2.rectangle(canvas, (b0, a0), (b1, a1), c)

    def wait(self, canvas: np.ndarray):
        fg = self.get_foreground()
        self.candidates[:] = 0
        for x in range(gsize):
            for y in range(gsize):
                a0, b0, a1, b1 = self.getrect(x, y)
                if (a1 - a0) * (b1 - b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255:
                    cv2.rectangle(canvas, (b0, a0), (b1, a1), (0, 0, 255))
