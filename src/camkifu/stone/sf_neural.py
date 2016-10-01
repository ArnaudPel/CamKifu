import cv2
import numpy as np

from camkifu.stone import StonesFinder
from camkifu.stone.nn_manager import NNManager, rcolors
from golib.config.golib_conf import gsize, E
from golib.model import Move
from golib.model.move import TK_TYPE, SGF_TYPE

__author__ = 'Arnaud Peloquin'


class SfNeural(StonesFinder):

    def __init__(self, vmanager):
        super().__init__(vmanager, learn_bg=True)
        self.manager = NNManager()
        self.candidates = np.zeros((gsize, gsize), dtype=np.uint8)
        self.indices = self.manager.class_indices()  # compute only once
        self.last_sugg = (-1, -1)
        self.has_sampled = False

    def _find(self, goban_img):
        #  todo this method is crying for a state machine
        if self.total_f_processed == 0:
            self.display_message("LOADING NEURAL NET...")
            self.manager.get_net()
        elif self.total_f_processed < self.bg_init_frames:
            self.display_message("BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames))
        elif not self.has_sampled:
            self.display_message("MAKING INITIAL ASSESSMENT...")
            print("initial assessment")
            self.predict_all(goban_img)
            self.has_sampled = True
        else:
            canvas = goban_img.copy()
            r, w, _ = goban_img.shape
            if r * w * 0.05 < np.sum(self.get_foreground()) / 255:
                self.wait(canvas)  # wait for things to settle
            else:
                self.search(goban_img, canvas=canvas)
            self._show(canvas, name='Neural')

    def predict_all(self, goban_img):
        stones = self.manager.predict_stones(goban_img)
        moves = []
        for i in range(gsize):
            for j in range(gsize):
                moves.append((stones[i][j], i, j))
        self.bulk_update(moves)

    def search(self, img, canvas: np.ndarray = None, color=(0, 255, 0)):
        fg = self.get_foreground()
        for r in range(gsize):
            for c in range(gsize):
                a0, b0, a1, b1 = self.getrect(r, c)
                if (a1 - a0) * (b1 - b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255:
                    self.candidates[r, c] += 1
                    if 2 < self.candidates[r, c] and (r, c) != self.last_sugg:
                        self.predict(r, c, img, canvas=canvas)
                        self.candidates[r, c] = 0
                    elif canvas is not None:
                        cv2.rectangle(canvas, (b0, a0), (b1, a1), color)

    def predict(self, r, c, img, canvas=None):
        """ Feed up to 4 input vectors to the neural network to predict the color of intersection (r, c)

        Since the neural network model takes images corresponding to 2x2 intersections, it is possible to
        extract 4 different images, in which the intersection of interest (r, c) takes respectively the position
        (3, 2, 1, 0) ~ (lower right, lower left, upper right, upper left)

        These prediction distributions are eventually reduced to predict the color at (r, c)
        """
        step = int((gsize + 1) / self.manager.split)
        x = []
        for k, (i, j) in enumerate(((-1, -1), (-1, 0), (0, -1), (0, 0))):
            rs = r + i
            cs = c + j
            if 0 <= rs < gsize - 1 and 0 <= cs < gsize - 1:
                x0, x1, y0, y1 = self.manager._get_rect_nn(rs, rs + step, cs, cs + step)
                pos = step ** 2 - k - 1  # remember the relative position of (r, c) in the sub-image
                x.append((pos, img[x0:x1, y0:y1]))
                if canvas is not None:
                    cv2.rectangle(canvas, (y0, x0), (y1, x1), color=(int(k * 255 / 3), 255, int((3-k) * 255 / 3)))
        y_s = self.manager.get_net().predict(np.asarray([img for _, img in x], dtype=np.float32))

        # reduce the distributions per color of the intersection (r, c)
        # todo is this the proper way to reduce softmax distributions in this case ? (mult, or train classifier ?)
        pred = np.zeros(3, dtype=np.float32)
        for i, y in enumerate(y_s):
            pos = x[i][0]  # the relative position (in the current sub-image) of the intersection (r, c)
            for color in range(3):
                pred[color] += np.sum(y[self.indices[pos, color]])
        new_color = rcolors[np.argmax(pred)]
        if new_color is not E:
            prev_color = self.get_stones()[r, c]
            if prev_color == E:
                self.suggest(new_color, r, c)
                self.last_sugg = (r, c)
            elif prev_color != new_color:
                loc = Move(TK_TYPE, (prev_color, r, c)).get_coord(SGF_TYPE)
                print("Err.. hum. Now seeing {} instead of {} at {}".format(new_color, prev_color, loc))

    def wait(self, canvas: np.ndarray):
        fg = self.get_foreground()
        self.candidates[:] = 0
        for x in range(gsize):
            for y in range(gsize):
                a0, b0, a1, b1 = self.getrect(x, y)
                if (a1 - a0) * (b1 - b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255:
                    cv2.rectangle(canvas, (b0, a0), (b1, a1), (0, 0, 255))
