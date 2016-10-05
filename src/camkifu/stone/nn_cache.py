import numpy as np

from golib.config.golib_conf import gsize

__author__ = 'Arnaud Peloquin'


class NNCache:

    def __init__(self, nn_manager, img):
        self.manager = nn_manager
        self.img = img  # todo  check img shape with manager
        self.cache = np.ndarray((self.manager.split, self.manager.split), dtype=object)
        self.step = (gsize + 1) // self.manager.split

    def predict_stone(self, r, c):
        i, j = self.get_area(r, c)
        y = self.predict_y(i, j)
        stones = self.manager.compute_stones(np.argmax(y))
        idx = self.step * (r % self.step) + c % self.step
        confidence = max(y) / sum(y)
        return stones[idx], confidence

    def predict_4_stones(self, i, j):
        y = self.predict_y(i, j)
        label = np.argmax(y)
        rs, re, cs, ce = self.manager._subregion(i, j)
        stones = self.manager.compute_stones(label).reshape((re - rs, ce - cs))
        confidence = max(y) / sum(y)
        return stones, confidence

    def predict_all_stones(self):
        stones = np.ndarray((gsize, gsize, 2), dtype=object)
        for i in range(self.manager.split):
            for j in range(self.manager.split):
                rs, re, cs, ce = self.manager._subregion(i, j)
                square, confidence = self.predict_4_stones(i, j)
                stones[rs:re, cs:ce, 0] = square
                stones[rs:re, cs:ce, 1] = confidence
        return stones

    def predict_y(self, i, j):
        """ A single 'y' label encodes a square of 2x2 intersections.

        """
        y = self.cache[i, j]
        if y is None:
            x = self._get_x(i, j)
            y = self.manager.get_net().predict(x.reshape(1, *x.shape))[0]
            self.cache[i, j] = y
        return y

    def predict_all_y(self):
        for i in range(self.manager.split):
            for j in range(self.manager.split):
                self.predict_y(i, j)
        return self.cache

    def _get_x(self, i, j):
        x0, x1, y0, y1 = self.manager._get_rect_nn(*self.manager._subregion(i, j))
        return self.img[x0:x1, y0:y1]

    def get_area(self, r, c):
        i = r // self.step
        j = c // self.step
        return i, j

