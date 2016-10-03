import functools
import math
import operator
import re as regex
from os.path import isfile

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from camkifu.config import cvconf
from camkifu.core.imgutil import show, draw_str, destroy_win
from golib.config.golib_conf import gsize, E, B, W
from golib.gui import ControllerBase

KERAS_MODEL_FILE = cvconf.train_dir + "/model/keras.h5"

__author__ = 'Arnaud Peloquin'

PNG_SUFFIX = ".png"
TRAIN_DAT_SUFFIX = '-train data.npz'
TRAIN_DAT_MTX = "all-train.npz"
colors = {E: 0, B: 1, W: 2}
rcolors = {0: E, 1: B, 2: W}


class NNManager:
    """The neural network Training Manager.

    """

    def __init__(self):
        # (~ to stonesfinder.canonical_shape)
        # all the size-related computations are based on this assumption
        self.canonical_shape = (cvconf.canonical_size, cvconf.canonical_size)
        self.depth = 3  # color channel

        self._network = None  # please access via self.get_net() - lazy instantiation
        self.split = 10  # feed the neural network with squares of four intersections (split the goban in 10x10 areas)
        self.nb_classes = 3 ** (int((gsize + 1) / self.split) ** 2)  # size of ANN output layer

        # find out the shape of such a square in order to get the number of features
        x0, x1, y0, y1 = self._get_rect_nn(*self._subregion(0, 0))
        self.r_width = y1 - y0  # number of pixels on the vertical side of the sample
        self.c_width = x1 - x0  # number of pixels on the horizontal side of the sample

        self.controller = ControllerBase()  # todo find a better way to load a saved game ? (this is using GUI package)
        self.c_indices = None  # see self.class_indices()

    def get_net(self):
        if self._network is None:
            self._network = self.init_net()
        return self._network

    def init_net(self):
        if isfile(KERAS_MODEL_FILE):
            print("Loading previous model from [{}] ...".format(KERAS_MODEL_FILE))
            model = load_model(KERAS_MODEL_FILE)
            print("... loading done")
            return model
        return self.create_net()

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
        assert img.shape[0:2] == self.canonical_shape
        stones = self.suggest_stones(img, img_path)
        if self.validate_stones(stones, img):
            return self.generate_xs(img), self.generate_ys(stones)
        else:
            return None, None

    def suggest_stones(self, img, img_path):
        game_path = NNManager.get_ref_game(img_path)
        y_path = NNManager.get_ref_y(img_path)
        if isfile(game_path):  # load stones from an sgf
            self.controller.loadkifu(game_path)
            self.controller.goto(999)
            stones = np.array(self.controller.rules.stones, dtype=object).T
        elif isfile(y_path):  # load stones from a previous numpy file
            stones = np.ndarray((gsize, gsize), dtype=object)
            y = np.load(y_path)['Y']
            for i in range(self.split):
                for j in range(self.split):
                    rs, re, cs, ce = self._subregion(i, j)
                    idx = i * self.split + j
                    s_arr = self.compute_stones(np.argmax(y[idx]))
                    stones[rs:re, cs:ce] = s_arr.reshape((re - rs, ce - cs))
            print("Loaded {}".format(y_path))
        else:  # try to guess stones with custom algo
            stones = self.predict_stones(img)
        return stones

    @staticmethod
    def get_ref_game(img_path):
        game_path = img_path.replace('snapshot', 'game').replace('.png', '.sgf')
        if isfile(game_path):
            return game_path
        return regex.sub(' \(\d*\)', '', game_path)

    @staticmethod
    def get_ref_y(path):
        y_path = path.replace(PNG_SUFFIX, '-y.npz')
        if isfile(y_path):
            return y_path
        # try to use a generic one if no specific has been found
        return regex.sub(' \(\d*\)', '', y_path)

    # noinspection PyBroadException
    def validate_stones(self, stones, img):
        # noinspection PyUnusedLocal
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
                        x_loc = int((x1 + x0) / 2) - 7
                        y_loc = int((y1 + y0) / 2) + 4
                        color = (255, 50, 50) if current == B else (0, 255, 255)
                        draw_str(canvas, current, x=x_loc, y=y_loc, color=color)
            show(canvas, name=win_name)
            cv2.setMouseCallback(win_name, onmouse)
            try:
                key = chr(cv2.waitKey(500))
            except:
                pass
            if key == 'r':
                stones[:] = E
                key = None
        destroy_win(win_name)
        return True if key == 'k' else False

    def generate_xs(self, img):
        x_s = np.zeros((self.split ** 2, self.r_width, self.c_width, self.depth), dtype=np.uint8)
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self._subregion(i, j)
                x0, x1, y0, y1 = self._get_rect_nn(rs, re, cs, ce)
                x_s[(i * self.split + j)] = img[x0:x1, y0:y1].copy()
        return x_s

    def generate_ys(self, stones):
        y_s = np.zeros((self.split ** 2, self.nb_classes), dtype=bool)
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self._subregion(i, j)
                label_val = self.compute_label(rs, re, cs, ce, stones)
                y_s[(i * self.split + j), label_val] = 1
        return y_s

    @staticmethod
    def compute_label(rs, re, cs, ce, stones):
        # todo refactor to take a 1-D collection of stones instead WARNING: check that np.flatten() is same as below !!
        label_val = 0
        for r in range(rs, re):
            for c in range(cs, ce):
                power = (r - rs) * (ce - cs) + (c - cs)
                label_val += colors[stones[r, c]] * (3 ** power)
        return label_val

    @staticmethod
    def compute_stones(label, dimension=4):
        k = label
        stones = np.ndarray(dimension, dtype=object)
        for i in reversed(range(dimension)):
            digit = int(k / (3**i))
            stones[i] = E if digit == 0 else B if digit == 1 else W
            k %= 3**i
        return stones

    def getrect(self, r, c, re=0, ce=0):
        x0 = int(r * self.canonical_shape[0] / gsize)
        y0 = int(c * self.canonical_shape[1] / gsize)
        re = (re + 1) if 0 < re else (r+1)
        ce = (ce + 1) if 0 < ce else (c+1)
        x1 = int(re * self.canonical_shape[0] / gsize)
        y1 = int(ce * self.canonical_shape[1] / gsize)
        return x0, y0, x1, y1

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

    def create_net(self):
        print("Creating new Keras model")
        network = Sequential()
        input_shape = (40, 40, self.depth)
        network.add(Convolution2D(32, 5, 5, activation='relu', input_shape=input_shape))
        network.add(Convolution2D(32, 5, 5, activation='relu'))
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(0.25))
        network.add(Convolution2D(90, 3, 3, activation='relu'))
        network.add(Convolution2D(90, 3, 3, activation='relu'))
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(0.25))
        network.add(Flatten())
        # Note: Keras does automatic shape inference.
        network.add(Dense(160, activation='relu'))
        network.add(Dropout(0.5))
        # output layer
        network.add(Dense(81, activation='softmax'))
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        network.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return network

    def train(self, x, y, batch_size=1000, nb_epoch=2, lr=0.001):
        assert len(x.shape) == 4  # expecting an array of colored images
        previous_lr = float(self.get_net().optimizer.lr.get_value())
        if not math.isclose(lr, previous_lr, rel_tol=1e-06):
            print("Setting new learning rate: {:.5f} -> {:.5f}".format(previous_lr, lr))
            self.get_net().optimizer.lr.set_value(lr)
        checkpoint = ModelCheckpoint(KERAS_MODEL_FILE, monitor='loss', save_best_only=True, save_weights_only=False)
        self.get_net().fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[checkpoint])

    def predict_ys(self, x):
        assert len(x.shape) == 4  # expecting an array of colored images
        return np.argmax(self.get_net().predict(x.astype(np.float32)), axis=1)

    def predict_stones(self, goban_img):
        x_s = self.generate_xs(goban_img)
        predict = self.predict_ys(x_s)
        stones = np.ndarray((gsize, gsize), dtype=object)
        for k, y in enumerate(predict):
            i = int(k / self.split)
            j = k % self.split
            rs, re, cs, ce = self._subregion(i, j)
            stones[rs:re, cs:ce] = self.compute_stones(y).reshape((re-rs, ce-cs))
        return stones

    def evaluate(self, x, y):
        assert len(x.shape) == 4  # expecting an array of colored images
        y_predict = self.get_net().predict(x.astype(np.float32))
        tp = 0  # true "positives" = non empty zones (at least one stone) detected correctly
        ap = 0  # all  "positives"
        tn = 0  # true "negatives"
        an = 0  # all  "negatives"
        for i, label in enumerate(y):
            pred = np.argmax(y_predict[i])
            truth = np.where(label == 1)[0]
            if 0 < truth:
                ap += 1
                if pred == truth:
                    tp += 1
                # else:
                #     print("Predicted {}, should be {}".format(pred, truth))
            else:
                an += 1
                if pred == truth:
                    tn += 1
                # else:
                #     print("Predicted {}, should be {}".format(pred, truth))
        assert ap + an == x.shape[0]
        tp_percent = 100 * tp / ap if 0 < ap else 0
        print('Non-empty: {:.2f} % ({}/{})'.format(tp_percent, tp, ap))
        tn_percent = 100 * tn / an if 0 < an else 0
        print('Empty    : {:.2f} % ({}/{})'.format(tn_percent, tn, an))

    def get_nb_weights(self):
        return sum([functools.reduce(operator.mul, layer.shape) for layer in self.get_net().get_weights()])

    def class_indices(self):
        """ Return an array of shape (dimension, nb_colors, n)
        where:
            - dimension is the number of intersections present in a subregions sent to neural training (typically 4)
            - nb_colors is the different values an intersection can take (E, B, W)
            - n is the number of classes matching these two keys (dimension, nb_colors)

        Eg. indices[0, 1] returns all the classes coding a Black stone at the first intersection (~ indices[0, B])
            indices[3, 2] returns all the classes coding a White stone at the forth intersection (~ indices[3, W])

        """
        if self.c_indices is None:
            nb_colors = len(colors)
            dimension = int(math.log(self.nb_classes, nb_colors))
            binar = np.zeros((self.nb_classes, dimension), dtype=object)
            self.c_indices = np.ndarray((dimension, nb_colors, int(self.nb_classes / nb_colors)), dtype=np.uint8)
            for c in range(self.nb_classes):
                binar[c] = NNManager.compute_stones(c, dimension=dimension)
            for d in range(dimension):
                for stone in colors.keys():
                    classes = np.where(binar[:, d] == stone)
                    self.c_indices[d, colors[stone]] = classes[0]
        return self.c_indices

    def patchwork(self, x, shape=(20, 20)):
        a, b = shape
        rw, cw = self.r_width, self.c_width
        idx = 0
        while True:
            canvas = np.zeros((a * rw, b * cw, self.depth), dtype=np.uint8)
            for j in range(a):
                for k in range(b):
                    if idx < len(x):
                        canvas[j*rw:(j+1)*rw, k*cw:(k+1)*cw] = x[idx]
                    else:
                        print("Images {} -> {}".format(idx - (j*b + k), idx))
                        yield canvas
                        return
                    idx += 1
            print("Images {} -> {}".format(idx - a*b, idx-1))
            yield canvas

    def visualize_inputs(self, x, shape=(20, 20)):
        for img in self.patchwork(x, shape):
            show(img, name='Patchwork')
            if chr(cv2.waitKey()) == 'q':
                break

    def visualize_l1(self):
        filters = self.get_net().get_weights()[0]
        nb_filt = filters.shape[3]
        nb_rows, nb_cols = 1, nb_filt
        for d in reversed(range(int(math.sqrt(nb_filt)))):
            if nb_filt % d == 0:
                nb_rows, nb_cols = d, nb_filt // d
                break
        depth = 3
        zoom = 8
        margin = 3
        img_height = (filters.shape[0] * zoom + margin) * nb_rows - margin
        img_width = (filters.shape[1] * zoom + margin) * nb_cols - margin
        canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        highs = []
        lows = []
        for color in range(depth):
            lows.append(np.min(filters[:, :, color, :]))
            highs.append(np.max(filters[:, :, color, :]))

        for i in range(nb_filt):
            f = filters[:, :, :, i]
            x0, y0 = (f.shape[0] * zoom + margin) * (i // nb_cols), (f.shape[1] * zoom + margin) * (i % nb_cols)
            x1, y1 = x0 + f.shape[0] * zoom, y0 + f.shape[1] * zoom
            for color in range(depth):
                f[:, :, color] = (f[:, :, color] - lows[color]) / (highs[color] - lows[color])
            canvas[x0:x1, y0:y1] = np.repeat(np.repeat(f * 255, zoom, axis=0), zoom, axis=1)
        show(canvas, name='Convolution filters - First layer')
        cv2.waitKey()
