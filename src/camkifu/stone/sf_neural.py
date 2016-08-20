import cv2
import numpy as np

from camkifu.core.imgutil import show, draw_str
from camkifu.stone import StonesFinder
from camkifu.stone.sf_clustering import SfClustering
from golib.config.golib_conf import gsize

__author__ = 'Arnaud Peloquin'


class SfNeural(StonesFinder):
    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.split = 10
        self.def_x = None
        self.def_y = None
        self.examples = None
        nb_labels = 3 ** (int((gsize + 1) / self.split) ** 2)
        self.labels = np.zeros((self.split ** 2, nb_labels), dtype=bool)

    def subregion(self, row, col):

        """ Get the row and column indices representing the requested subregion of the goban.

        Ex if split is 3, the region (0, 1) has row indices (0, 6), and column indices (6, 12).

        Note: the border regions (end of row or col) cross over the previous ones if need be,
        so that the shape of each subregion is always the same.

        Args:
            row: int
            col: int
                The identifiers of the region, in [ 0, self.split [

        Returns rs, re, cs, ce:  int, int, int, int
            rs -- row start     (inclusive)
            re -- row end       (exclusive)
            cs -- column start  (inclusive)
            ce -- column end    (exclusive)
        """
        assert 0 <= row < self.split and 0 <= col < self.split
        step = int((gsize + 1) / self.split)
        rs = row * step
        re = (row + 1) * step
        if gsize - rs < step:
            rs = gsize - step
            re = gsize

        cs = col * step
        ce = (col + 1) * step
        if gsize - cs < step:
            cs = gsize - step
            ce = gsize

        return rs, re, cs, ce

    def gen_data(self):
        img = cv2.imread("/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/gobanimg 1.png")
        clustering = SfClustering(None)
        stones = clustering.find_stones(img)
        # canvas = self.draw_stones(stones)
        # show(canvas, name="Kmeans")
        # cv2.waitKey()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.parse_regions(gray, stones)

    def parse_regions(self, img, stones):
        key = None
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self.subregion(i, j)
                x0, x1, y0, y1 = self.get_rect_nn(rs, re, cs, ce)
                subimg = img[x0:x1, y0:y1].copy()
                label_val = 0
                for r in range(rs, re):
                    for c in range(cs, ce):
                        x = int((c - cs + 0.5) * self.def_x / (ce - cs))
                        y = int((r - rs + 0.5) * self.def_y / (re - rs))
                        stone = stones[r, c]
                        draw_str(subimg, stone, x=x - 4, y=y + 6)
                        digit = 0 if stone is 'E' else 1 if stone is 'B' else 2
                        power = (r - rs) * (ce - cs) + (c - cs)
                        label_val += digit * (3 ** power)
                show(subimg)
                key = None
                try:
                    key = chr(cv2.waitKey())
                except:
                    pass
                if key is 'q':
                    print('ok bye')
                    break
                if key is 'y' or key is 'o':
                    example_idx = i * self.split + j
                    self.examples[example_idx] = subimg.flatten()
                    self.labels[example_idx, label_val] = 1
                    print("sample {} is {} (X sum={})".format(example_idx, label_val, np.sum(self.examples)))
            if key is 'q':
                break

    def get_rect_nn(self, rs, re, cs, ce):
        """ For machine learning (neural networks) data generation.
        """
        x0, y0, _, _ = self.getrect(rs, cs)
        _, _, x1, y1 = self.getrect(re - 1, ce - 1)
        if self.def_x is None:
            self.def_x = x1 - x0
            self.def_y = y1 - y0
            self.examples = np.zeros((self.split**2, self.def_x * self.def_y))
        if x1 - x0 != self.def_x:
            x0 = x1 - self.def_x
        if y1 - y0 != self.def_y:
            y0 = y1 - self.def_y
        return x0, x1, y0, y1


if __name__ == '__main__':
    sf = SfNeural(None)
    sf.gen_data()
