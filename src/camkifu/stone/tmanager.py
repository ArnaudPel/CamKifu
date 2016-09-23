from os import listdir

import cv2
import numpy as np
from os.path import join, isfile

import re as regex

from keras import backend
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from camkifu.config import cvconf
from camkifu.core.imgutil import show, draw_str, destroy_win
from camkifu.stone.sf_clustering import SfClustering
from golib.config.golib_conf import gsize, E, B, W
from golib.gui import ControllerBase

KERAS_MODEL_FILE = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/keras/model.h5"

__author__ = 'Arnaud Peloquin'

PNG_SUFFIX = ".png"
TRAIN_DAT_SUFFIX = '-train data.npz'
TRAIN_DAT_MTX = "all-train.npz"


class TManager:
    """The neural network Training Manager.

    """

    def __init__(self):
        # (~ to stonesfinder.canonical_shape)
        # all the size-related computations are based on this assumption
        self.canonical_shape = (cvconf.canonical_size, cvconf.canonical_size)

        self._network = None  # please access via self.get_net() - lazy instantiation
        self.split = 10  # feed the neural network with squares of four intersections (split the goban in 10x10 areas)
        self.nb_classes = 3 ** (int((gsize + 1) / self.split) ** 2)  # size of ANN output layer

        # find out the shape of such a square in order to get the number of features
        x0, x1, y0, y1 = self._get_rect_nn(*self._subregion(0, 0))
        self.r_width = y1 - y0  # number of pixels on the vertical side of the sample
        self.c_width = x1 - x0  # number of pixels on the horizontal side of the sample

        self.controller = ControllerBase()  # todo find a better way to load a saved game ? (this is using GUI package)

    def get_net(self):
        if self._network is None:
            self._network = TManager.init_net()
        return self._network

    @staticmethod
    def init_net():
        if isfile(KERAS_MODEL_FILE):
            print("Loading previous model from [{}] ...".format(KERAS_MODEL_FILE))
            model = load_model(KERAS_MODEL_FILE)
            print("... loading done")
            return model
        return TManager.create_net()

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
            return self._label_regions(img, stones)
        else:
            return None, None

    def suggest_stones(self, img, img_path):
        game_path = TManager.get_ref_game(img_path)
        y_path = TManager.get_ref_y(img_path)
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
                    example_idx = i * self.split + j
                    slist = self.compute_stones(np.argmax(y[example_idx]))
                    stones[rs:re, cs:ce] = np.array(slist).reshape((re - rs, ce - cs))
            print("Loaded {}".format(y_path))
        else:  # try to guess stones with custom algo
            stones = SfClustering(None).find_stones(img)
        return stones

    @staticmethod
    def get_ref_game(img_path):
        game_path = img_path.replace('snapshot', 'game').replace('.png', '.sgf')
        return regex.sub(' \(\\d*\)', '', game_path)

    @staticmethod
    def get_ref_y(path):
        return path.replace(PNG_SUFFIX, '-y.npz')

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

    def _label_regions(self, img, stones):
        examples = np.zeros((self.split ** 2, self.r_width, self.c_width, 3), dtype=np.uint8)
        labels = np.zeros((self.split ** 2, self.nb_classes), dtype=bool)
        for i in range(self.split):
            for j in range(self.split):
                rs, re, cs, ce = self._subregion(i, j)
                x0, x1, y0, y1 = self._get_rect_nn(rs, re, cs, ce)
                example_idx = i * self.split + j
                examples[example_idx] = img[x0:x1, y0:y1].copy()
                label_val = self.compute_label(rs, re, cs, ce, stones)
                labels[example_idx, label_val] = 1
        return examples, labels

    @staticmethod
    def compute_label(rs, re, cs, ce, stones):
        # todo refactor to take a 1-D collection of stones instead WARNING: check that np.flatten() is same as below !!
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

    @staticmethod
    def create_net():
        print("Creating new Keras model")
        network = Sequential()
        input_shape = (40, 40, 3)
        network.add(Convolution2D(32, 3, 3, activation='relu', border_mode='valid', input_shape=input_shape))
        network.add(Convolution2D(32, 3, 3, activation='relu'))
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(0.25))
        network.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid'))
        network.add(Convolution2D(64, 3, 3, activation='relu'))
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(0.25))
        network.add(Flatten())
        # Note: Keras does automatic shape inference.
        network.add(Dense(160, activation='relu'))
        network.add(Dropout(0.5))
        # output layer
        network.add(Dense(81, activation='softmax'))
        sgd = SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
        network.compile(loss='categorical_crossentropy', optimizer=sgd)
        return network

    def train(self, x, y, batch_size=1000, nb_epoch=2):
        assert len(x.shape) == 4  # expecting an array of colored images
        self.get_net().optimizer.lr = backend.variable(0.003)
        print('Starting CNN training..')
        self.get_net().fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch)
        self.get_net().save(KERAS_MODEL_FILE)
        print('.. CNN training done')

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
        print('Non-empty: {:.2f} % ({}/{})'.format(100 * tp / ap, tp, ap))
        print('Empty    : {:.2f} % ({}/{})'.format(100 * tn / an, tn, an))

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

    @staticmethod
    def display_histo(labels, nb_bins=3 ** 4):
        """
        labels -- the label vectors as fed to the neural network
        """
        histo = TManager.raw_histo(labels, nb_bins)
        cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
        zoom = 7
        chart_width = nb_bins * zoom
        bins = np.arange(0, chart_width, zoom).reshape(nb_bins, 1)
        pts = np.column_stack((bins, (np.int32(np.around(histo)))))
        canvas = np.zeros((300, chart_width, 3))
        cv2.polylines(canvas, [pts], False, (255, 255, 255))
        win_name = 'Training data labels distribution'
        show(np.flipud(canvas), name=win_name)
        cv2.waitKey()
        destroy_win(win_name)

    @staticmethod
    def raw_histo(labels, nb_bins=3 ** 4):
        y = np.zeros((labels.shape[0], 1), dtype=np.uint8)
        for i, label_vect in enumerate(labels):
            y[i] = np.argmax(label_vect)
        return cv2.calcHist([y], [0], None, [nb_bins], [0, nb_bins])


def train_eval(manager, base_dir):
    x_train, y_train, x_eval, y_eval = split_data(base_dir, manager)
    for _ in range(30):
        manager.train(x_train, y_train, batch_size=920, nb_epoch=2)
    manager.evaluate(x_eval, y_eval)


def split_data(base_dir, manager):
    tfile = join(base_dir, TRAIN_DAT_MTX)
    if isfile(tfile):
        tdat = np.load(tfile)
        xt, yt = tdat['Xt'], tdat['Yt']
        xe, ye = tdat['Xe'], tdat['Ye']
        print("Loaded previous datasets from [{}]".format(TRAIN_DAT_MTX))
    else:
        x_tot, y_tot = manager.merge_trains(base_dir)
        indices = np.indices([x_tot.shape[0]])[0]
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        xt = x_tot[indices[:split_idx], :]
        yt = y_tot[indices[:split_idx], :]
        xe = x_tot[indices[split_idx:], :]
        ye = y_tot[indices[split_idx:], :]
        np.savez(tfile, Xt=xt, Yt=yt, Xe=xe, Ye=ye)
        print("Created NEW datasets: [{}]".format(TRAIN_DAT_MTX))
    return xt, yt, xe, ye


def extract_ys(base_dir):
    """ Extract the Y matrix from every train-data file for which no ref game can be found.

    """
    for mat in [f for f in listdir(base_dir) if f.endswith(PNG_SUFFIX)]:
        path = join(base_dir, mat)
        if not isfile(TManager.get_ref_game(path)):
            dat_path = path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)
            if isfile(dat_path):
                np.savez(TManager.get_ref_y(path), Y=np.load(dat_path)['Y'])


if __name__ == '__main__':
    manager = TManager()
    base_dir = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/"

    train_eval(manager, base_dir)
    # extract_ys(base_dir)
