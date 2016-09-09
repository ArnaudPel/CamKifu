from os import listdir

import cv2
import numpy as np
from os.path import join

from camkifu.core.imgutil import show, draw_str, destroy_win
from camkifu.stone import StonesFinder
from camkifu.stone.sf_clustering import SfClustering
from golib.config.golib_conf import gsize, E, B, W

__author__ = 'Arnaud Peloquin'

TRAIN_DAT_SUFFIX = '-train data.npz'


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
        stones = self.suggest_stones(img)
        return self._label_regions(img, stones) if stones is not None else None

    def suggest_stones(self, img):
        stones = SfClustering(None).find_stones(img)

        def onmouse(event, x, y, flag, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                row = int(gsize * y / self.canonical_shape[1])
                col = int(gsize * x / self.canonical_shape[0])
                prev = stones[row, col]
                if prev != E:
                    stones[row, col] = E
                else:
                    stones[row, col] = B if key == 'b' else W

        key = None
        win_name = 'Labels Validation'
        while key not in ('k', 'q'):
            canvas = img.copy()
            for r in range(gsize):
                for c in range(gsize):
                    x0, y0, x1, y1 = self.getrect(r, c)
                    current = stones[c, r]
                    if current in (B, W):
                        x_loc = int((x1 + x0) / 2) - 4
                        y_loc = int((y1 + y0) / 2) + 6
                        color = (255, 50, 50) if current == B else (0, 255, 255)
                        draw_str(canvas, current, x=x_loc, y=y_loc, color=color)
            show(canvas, name=win_name)
            cv2.setMouseCallback(win_name, onmouse)
            try:
                key = chr(cv2.waitKey(500))
            except:
                pass
            if key == 'r':
                stones = np.zeros((gsize, gsize), dtype=object)
                stones[:] = E
                key = None

        if key == 'q':
            stones = None
        destroy_win(win_name)
        return stones

    def _label_regions(self, img, stones):
        examples = np.zeros((self.split ** 2, self.nb_features), dtype=np.uint8)
        labels = np.zeros((self.split ** 2, self.nb_classes), dtype=bool)
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self._subregion(i, j)
                x0, x1, y0, y1 = self._get_rect_nn(rs, re, cs, ce)
                example_idx = i * self.split + j

                # feed grayscale to nn as a starter. todo use colors ? (3 times as more input nodes)
                examples[example_idx] = cv2.cvtColor(img[x0:x1, y0:y1], cv2.COLOR_BGR2GRAY).flatten()
                label_val = self.compute_label(ce, cs, re, rs, stones)
                labels[example_idx, label_val] = 1
                print("sample {} is {} {}".format(example_idx, label_val, self.compute_stones(label_val)))
        return examples, labels

    @staticmethod
    def compute_label(ce, cs, re, rs, stones):
        label_val = 0
        for r in range(rs, re):
            for c in range(cs, ce):
                stone = stones[r, c]
                digit = 0 if stone == E else 1 if stone == B else 2
                power = (r - rs) * (ce - cs) + (c - cs)
                label_val += digit * (3 ** power)
        return label_val

    @staticmethod
    def compute_stones(label, dimension=4):
        k = label
        stones = []
        for i in reversed(range(dimension)):
            digit = int(k / (3**i))
            stones.append(E if digit == 0 else B if digit == 1 else W)
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

    def train(self, train_data_file):
        file = np.load(train_data_file)
        x = file['X']
        y = file['Y']
        print('Loaded {}'.format(train_data_file))
        print('Starting ANN training..')
        self.neurons.setLayerSizes(np.int32([x.shape[1], 100, 100, y.shape[1]]))
        self.neurons.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.neurons.setBackpropMomentumScale(0.0)
        self.neurons.setBackpropWeightScale(0.001)
        self.neurons.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.neurons.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.neurons.train(x.astype(np.float32), cv2.ml.ROW_SAMPLE, y.astype(np.float32))
        # https://github.com/opencv/opencv/issues/4969
        # self.neurons.save(train_data_path.replace('-train data.npz', '-neurons.yml'))
        print('.. training done')

    def predict(self, test_file):
        file = np.load(test_file)
        x = file['X']
        y = file['Y']
        print('Loaded ' + test_file)
        ret, y_predict = self.neurons.predict(x.astype(np.float32))
        count = 0
        for i, label in enumerate(y):
            pred = np.argmax(y_predict[i])
            truth = np.where(label == 1)[0]
            if pred == truth:
                count += 1
            else:
                print("Predicted {}, should be {}".format(pred, truth))
        print('Accuracy: {} %'.format(100 * count / x.shape[0]))

    @staticmethod
    def merge_trains(train_data_dir):
        inputs = []
        labels = []
        for mat in [f for f in listdir(train_data_dir) if f.endswith(TRAIN_DAT_SUFFIX)]:
            data = np.load(join(train_data_dir, mat))
            inputs.append(data['X'])
            labels.append(data['Y'])
        x = np.concatenate(inputs, axis=0)
        print('Merged {} inputs {} -> {}'.format(len(inputs), inputs[0].shape, x.shape))
        y = np.concatenate(labels, axis=0)
        print('Merged {} labels {} -> {}'.format(len(labels), labels[0].shape, y.shape))
        return x, y

if __name__ == '__main__':
    sf = SfNeural(None, )
    base_dir = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/"
    img = "{}snapshot-1.png".format(base_dir)

    # X, Y = sf.gen_data(img)
    # np.savez(img.replace(".png", TRAIN_DAT_SUFFIX), X=X, Y=Y)

    sf.train('/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/all train.npz')
    sf.predict('/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/snapshot-13-cross-valid data.npz')
