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
        self.neurons = cv2.ml.ANN_MLP_create()
        self.split = 10  # feed the neural network with squares of four intersections (split the goban in 10x10 areas)

        # find out the shape of such a square in order to get the number of features
        x0, x1, y0, y1 = self._get_rect_nn(*self._subregion(0, 0))
        self.r_width = y1 - y0  # number of pixels on the vertical side of the sample
        self.c_width = x1 - x0  # number of pixels on the horizontal side of the sample

        self.nb_features = self.r_width * self.c_width               # size of ANN input layer (nb of pixels)
        self.nb_classes = 3 ** (int((gsize + 1) / self.split) ** 2)  # size of ANN output layer

    def _subregion(self, row, col):

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

    def gen_data(self, img_path):
        img = cv2.imread(img_path)
        assert img.shape[0:2] == self.canonical_shape  # all the size computations are based on this assumption
        clustering = SfClustering(None)
        stones = clustering.find_stones(img)
        # canvas = self.draw_stones(stones)
        # show(canvas, name="Kmeans")
        # cv2.waitKey()
        return self._parse_regions(img, stones)

    def _parse_regions(self, img, stones):
        examples = np.zeros((self.split ** 2, self.nb_features), dtype=np.uint8)
        labels = np.zeros((self.split ** 2, self.nb_classes), dtype=bool)
        break_keys = ('q', 'y')
        adjust_keys = ('e', 'b', 'w')
        key = None
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self._subregion(i, j)
                x0, x1, y0, y1 = self._get_rect_nn(rs, re, cs, ce)
                for r in range(rs, re):
                    for c in range(cs, ce):
                        subimg = img[x0:x1, y0:y1].copy()
                        self.draw_suggestion(rs, re, cs, ce, stones, subimg)
                        a0, b0, a1, b1 = self.getrect(r, c)
                        cv2.rectangle(subimg, (b0 - y0, a0 - x0), (b1 - y0, a1 - x0), (0, 0, 255), thickness=2)
                        show(subimg)
                        try:
                            key = chr(cv2.waitKey())
                            while key not in break_keys + adjust_keys:
                                print("unknown command. please use: {} or fix with {}".format(break_keys, adjust_keys))
                                key = chr(cv2.waitKey())
                        except Exception as e:
                            print(e)
                            key = 'q'
                        if key in break_keys:
                            break
                        elif key in adjust_keys:
                            stones[r, c] = key.upper()
                    if key in break_keys:
                        break

                if key not in ('q', None):
                    example_idx = i * self.split + j
                    # feed grayscale to nn as a starter. todo use colors ? (3 times as more input nodes)
                    examples[example_idx] = cv2.cvtColor(img[x0:x1, y0:y1], cv2.COLOR_BGR2GRAY).flatten()
                    label_val = self.compute_label(ce, cs, re, rs, stones)
                    labels[example_idx, label_val] = 1
                    print("sample {} is {} {}".format(example_idx, label_val, self.compute_stones(label_val)))
                else:
                    break
            if key is 'q':
                print('bye bye')
                break
        return examples, labels

    def draw_suggestion(self, rs, re, cs, ce, stones, subimg):
        for r in range(rs, re):
            for c in range(cs, ce):
                x = int((c - cs + 0.5) * self.c_width / (ce - cs))
                y = int((r - rs + 0.5) * self.r_width / (re - rs))
                stone = stones[r, c]
                draw_str(subimg, stone, x=x - 4, y=y + 6)

    def compute_label(self, ce, cs, re, rs, stones):
        label_val = 0
        for r in range(rs, re):
            for c in range(cs, ce):
                stone = stones[r, c]
                digit = 0 if stone == 'E' else 1 if stone == 'B' else 2
                power = (r - rs) * (ce - cs) + (c - cs)
                label_val += digit * (3 ** power)
        return label_val

    def compute_stones(self, label, dimension=4):
        k = label
        stones = []
        for i in reversed(range(dimension)):
            digit = int(k / (3**i))
            stones.append('E' if digit == 0 else 'B' if digit == 1 else 'W')
            k %= 3**i
        return list(reversed(stones))

    def _get_rect_nn(self, rs, re, cs, ce):
        """ For machine learning (neural networks) data generation.
        """
        x0, y0, _, _ = self.getrect(rs, cs)
        _, _, x1, y1 = self.getrect(re - 1, ce - 1)
        if hasattr(self, 'c_width'):
            if x1 - x0 != self.c_width:
                x0 = x1 - self.c_width
            if y1 - y0 != self.r_width:
                y0 = y1 - self.r_width
        return x0, x1, y0, y1

    def train(self, train_data_path):
        file = np.load(train_data_path)
        x = file['X']
        y = file['Y']
        print('Loaded {}' + train_data_path)
        self.neurons.setLayerSizes(np.int32([x.shape[1], 100, 100, y.shape[1]]))
        self.neurons.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.neurons.setBackpropMomentumScale(0.0)
        self.neurons.setBackpropWeightScale(0.001)
        self.neurons.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.neurons.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.neurons.train(x.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.float32))

        # https://github.com/opencv/opencv/issues/4969
        # self.neurons.save(train_data_path.replace('-train data.npz', '-neurons.yml'))

        print('Trained neural network')

    def predict(self, train_data_path):
        file = np.load(train_data_path)
        x = file['X']
        y = file['Y']
        print('Loaded ' + train_data_path)
        ret, y_predict = self.neurons.predict(x)
        for i, label in enumerate(y):
            pred = np.argmax(y_predict[i])
            truth = np.where(label == 1)[0]
            print("Predicted {}, should be {}".format(pred, truth))

if __name__ == '__main__':
    sf = SfNeural(None, )
    img_path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/snapshot-1.png"
    examples, labels = sf.gen_data(img_path)
    np.savez(img_path.replace('.png', '-train data.npz'), X=examples, Y=labels)  # X and Y are the matrices names
