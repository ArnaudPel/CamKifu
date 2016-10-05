import math

import cv2
import numpy as np

from camkifu.stone import StonesFinder
from camkifu.stone.nn_cache import NNCache
from camkifu.stone.nn_manager import NNManager
from golib.config.golib_conf import gsize, E, B, W
from golib.model import Move
from golib.model.move import NP_TYPE, KGS_TYPE

MIN_CONFIDENCE = 0.6

COLD = 'cold'
WIN_NAME = 'Neural'

TARGET_THRESH = 15
TARGET_INCR = 5
NB_LOOKBACK = 3

__author__ = 'Arnaud Peloquin'


class SfNeural(StonesFinder):

    def __init__(self, vmanager):
        super().__init__(vmanager, learn_bg=True)
        self.manager = NNManager()
        self.cache = None
        self.targets = np.zeros((gsize, gsize), dtype=np.uint8)
        self.indices = self.manager.class_indices()  # compute only once
        self.has_sampled = False
        self.heatmap = np.ndarray((gsize, gsize), dtype=object)

    def _find(self, goban_img):
        self.cache = NNCache(self.manager, goban_img)
        #  todo this method is crying for a state machine
        if self.total_f_processed == 0:
            self.display_message("LOADING NEURAL NET...", name=WIN_NAME)
            self.manager.get_net()
        elif self.total_f_processed < self.bg_init_frames:
            bg_message = "BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames)
            self.display_message(bg_message, name=WIN_NAME)
        elif not self.has_sampled:
            self.display_message("MAKING INITIAL ASSESSMENT...", name=WIN_NAME, force=True)
            print("initial assessment")
            self.predict_all()
            self.has_sampled = True
        else:
            canvas = goban_img.copy()
            self.mark_targets(canvas)
            self.process_targets(canvas)
            self.lookback(canvas=canvas)
            self._show(canvas, name=WIN_NAME)

    def predict_all(self):
        stones = self.cache.predict_all_stones()
        moves = []
        discarded = []
        for r in range(gsize):
            for c in range(gsize):
                color, confidence = stones[r, c]
                if color is not E:
                    if confidence > MIN_CONFIDENCE:
                        moves.append((color, r, c))
                        self.heatmap[r, c] = HeatPoint(color, confidence, self.total_f_processed)
                    else:
                        discarded.append((color, r, c))
        print('predict_all: discarded {}'.format(discarded))
        self.bulk_update(moves)

    def mark_targets(self, canvas):
        fg = self.get_foreground()
        for r in range(gsize):
            for c in range(gsize):
                if self.heatmap[r, c] is None:
                    if self.is_agitated(r, c, fg):
                        self.targets[r, c] += TARGET_INCR
                    if self.targets[r, c]:
                        x0, y0, x1, y1 = self.getrect(r, c)
                        red = min(255, int(self.targets[r, c]) / TARGET_INCR * 255 // 4)
                        cv2.rectangle(canvas, (y0, x0), (y1, x1), (0, 0, red))
        self.targets[np.where(self.targets > 0)] -= 1  # decay

    def process_targets(self, canvas):
        targets = self.select_targets(canvas)
        moves = self.predict_moves(targets)
        if len(moves):
            if self.get_color_ratio(moves) < 1:
                for (color, r, c, confidence) in moves:
                    self.heatmap[r, c] = HeatPoint(color, confidence, self.total_f_processed)
                if len(moves) == 1:
                    self.suggest(*moves.pop()[0:3])
                else:
                    self.bulk_update([m[0:3] for m in moves])

    def predict_moves(self, targets):
        """
        Args:
            targets: iterable
                The coordinates (i, j) of the target zones.
        """
        moves = set()
        if not len(targets):
            return moves
        stones = self.get_stones()
        for i, j in targets:
            new_stones, confidence = self.cache.predict_4_stones(i, j)
            if confidence < MIN_CONFIDENCE:
                continue
            rs, re, cs, ce = self.manager._subregion(i, j)
            for a, b in np.transpose(np.where(new_stones != E)):
                r = a + rs
                c = b + cs
                prev_color = stones[r, c]
                new_color = new_stones[a, b]
                if prev_color == E:
                    moves.add((new_color, r, c, confidence))
                elif prev_color != new_color:
                    loc = Move(NP_TYPE, (prev_color, r, c)).get_coord(KGS_TYPE)
                    print("Err.. hum. Now seeing {} instead of {} at {}".format(new_color, prev_color, loc))
        return moves

    def select_targets(self, canvas):
        """ Select marked targets that are not too agitated (as in background/foreground separation)

        """
        fg = self.get_foreground()
        targets = []
        for i in range(self.manager.split):
            for j in range(self.manager.split):
                rs, re, cs, ce = self.manager._subregion(i, j)
                agitated = False
                if not len(np.where(self.targets[rs:re, cs:ce] > TARGET_THRESH)[0]):
                    continue
                for a in range(rs, re):
                    for b in range(cs, ce):
                        # todo set a different agitation threshold for target processing than selection
                        if self.is_agitated(a, b, fg):
                            agitated = True
                            if agitated: break
                    if agitated: break
                x0, x1, y0, y1 = self.manager._get_rect_nn(rs, re, cs, ce)
                if not agitated:
                    targets.append((i, j))
                    self.targets[rs:re, cs:ce] = 0
                    cv2.rectangle(canvas, (y0, x0), (y1, x1), (0, 255, 0))
                else:
                    cv2.rectangle(canvas, (y0, x0), (y1, x1), (255, 0, 0))
        return targets

    def lookback(self, canvas=None):
        """ Re-check recent predictions, and cancel them if they are not consistent.

        """
        stones = self.get_stones()
        to_del = []
        for r, c in np.transpose(np.where(self.heatmap == HeatPoint)):  # trick to express 'where not None'
            hpoint = self.heatmap[r, c]
            if hpoint.color != stones[r, c]:
                self.heatmap[r, c] = None  # the location has been modified by someone else, leave it alone
                continue
            if 10 < self.total_f_processed - hpoint.stamp:
                hpoint.stamp = self.total_f_processed
                hpoint.check(*self.cache.predict_stone(r, c))
                if not hpoint.is_valid():
                    to_del.append((E, r, c))  # cancel previous prediction
        if len(to_del):
            self.bulk_update(to_del)
        self._cleanup_heatmap()
        if canvas is not None:
            self._drawvalues(canvas, np.transpose(self.heatmap))

    def is_agitated(self, r, c, fg):
        a0, b0, a1, b1 = self.getrect(r, c)
        return (a1 - a0) * (b1 - b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255

    def _cleanup_heatmap(self):
        for r, c in np.transpose(np.where(self.heatmap == COLD)):
            self.heatmap[r, c] = None

    @staticmethod
    def get_color_ratio(moves):
        count = {B: 0, W: 0}
        for m in moves:
            if m[0] != E:
                count[m[0]] += 1
        if 0 in count.values():
            count[B] += 1
            count[W] += 1
        return abs(math.log(count[B] / count[W], 3))


class HeatPoint:

    def __init__(self, color, confidence, stamp, energy=NB_LOOKBACK):
        self.target = energy
        self.energy = energy
        self.color = color
        self.confidence = confidence
        self.stamp = stamp
        self.nb_checks = 0
        self.nb_passed = 0

    def check(self, color, confidence: float):
        self.nb_checks += 1
        self.energy -= 1
        new_conf = 0
        if color == self.color:
            self.nb_passed += 1
            new_conf = confidence
        # compute confidence after increment of self.nb_checks in order to count the initial confidence weight
        self.confidence = (self.confidence * self.nb_checks + new_conf) / (self.nb_checks + 1)

    def is_valid(self):
        # fail as soon as energy no longer allows to reach target
        can_pass = 2 * self.target / 3 <= self.nb_passed + self.energy
        if not can_pass:
            self.energy = 0  # no need to check this location any longer
            self.confidence = 0.0
        return can_pass

    def is_cold(self):
        return self.energy < -5

    def __eq__(self, *args, **kwargs):
        if len(args):
            if args[0] is HeatPoint:
                return 0 < self.energy
            if args[0] == COLD:
                return self.is_cold()
        return super().__eq__(*args, **kwargs)

    def __repr__(self):
        if self.energy <= 0:
            self.energy -= 1
        if not self.is_cold():
            return '{:d}'.format(int(self.confidence*10))
        else:
            return ''
