from math import pi
from queue import Empty

import cv2
from numpy import zeros_like, zeros, uint8, int32, float32, empty, empty_like, sum as npsum, array, max as npmax,\
    min as npmin, mean as npmean
from numpy.ma import absolute
from time import time
import sys

from camkifu.core.imgutil import draw_str, draw_contours_multicolor, connect_clusters, sort_contours_box
from golib.model.move import Move
from camkifu.stone.stonesfinder import StonesFinder, compare, evalz
from golib.config.golib_conf import gsize, B, W, E


__author__ = 'Arnaud Peloquin'

# the number of background sampling frames before allowing stones detection (background learning phase).
bg_learning_frames = 50

# possible states for a BackgroundSub2 instance :
sampling = "sampling"
# watching = "watching"
searching = "searching"

# number of frames to accumulate before running 'statistics' to find a stone
accumulation_passes = 7


class BackgroundSub2(StonesFinder):
    """
    Save background data using sample(img).
    Perform background subtraction operations in order to detect stones.

    """

    label = "Bg Sub 2"

    def __init__(self, vmanager):
        super(BackgroundSub2, self).__init__(vmanager)

        # doc : cv2.BackgroundSubtractor.apply(image[, fgmask[, learningRate]]) → fgmask
        self._bg_model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._bg_initialization = 0
        self.background = None
        self.fg_mask = None
        self.candidates = []  # candidates for the next added stone
        self.candid_acc = 0  # the number of frames since last candidate list clear

        self.intersections = zeros((19, 19), dtype=object)  # keep track of intersections that have been moving
        self.lastpos = None

        self.state = sampling
        self.nb_untouched = 0  # the number of successive searches that detected no motion at all
        self.last_on = time()  # instant when last active. to be used to detect long sleeps.

        self.total_f_processed = 0  # total number of frames processed since init. Dev var essentially.

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)  # todo search what's best here given the new bg modeling
        if self.state == sampling:
            done = self.sample(filtered)
            if done:
                self.state = searching
        else:
            self.search(filtered)
            self.last_on = time()
        self.total_f_processed += 1
        # self._show(filtered)

    def _learn(self):
        try:
            while True:
                # todo implement correction in case of deletion by user (see base method doc).
                err, exp = self.corrections.get_nowait()
                print("%s has become %s" % (err, exp))
        except Empty:
            pass

    def sample(self, img):
        """
        Return True when enough images have been applied to the background model and it's considered ready to use.

        """
        if self.background is None:
            self.background = array(img, dtype=float32)
        cv2.accumulateWeighted(img, self.background, 0.01)
        if self.fg_mask is None:
            self.fg_mask = zeros((img.shape[0], img.shape[1]), dtype=uint8)
        self._bg_model.apply(img, fgmask=self.fg_mask, learningRate=0.01)
        self._bg_initialization += 1
        if self._bg_initialization < bg_learning_frames:
            black = zeros((img.shape[0], img.shape[1]), dtype=uint8)
            draw_str(black, int(black.shape[0] / 2 - 70), int(black.shape[1] / 2),
                     "SAMPLING ({0}/{1})".format(self._bg_initialization, bg_learning_frames))
            self._show(black)
            return False
        return True

    def get_intersection(self, x, y):
        inter = self.intersections[x][y]
        if inter == 0:
            # see numpy.vectorize(Intersection) to instanciate a full grid of objects
            inter = Intersection()
            self.intersections[x][y] = inter
        return inter

    def search(self, img):
        """
        Try to detect stones by analysing foreground mask and comparing ROIs with accumulated background.

        """
        cv2.accumulateWeighted(img, self.background, 0.01)  # todo see to put the inverted foreground as mask ?
        expected_radius = max(*img.shape) / gsize / 2  # the approximation of the radius of a stone, in pixels

        # todo read paper about MOG2, in order to know how to use it properly here
        learn = 0 if self.total_f_processed % 5 else 0.01
        fg = self._bg_model.apply(img, fgmask=self.fg_mask, learningRate=learn)
        sorted_conts, contours = self.extract_contours(fg, expected_radius)
        # colors = zeros_like(img)
        # draw_contours_multicolor(colors, contours)

        # search for a contour that could be a new stone. the unlikely areas have been trimmed already.
        # roi_mask = zeros_like(img)  # regions of interest
        for wrapper in sorted_conts:
            # todo extract constraint-checking methods so that they can be re-ordered, and also just to clean the code
            # the bounding box must be a rough square
            if 2 / 3 < wrapper.box[1][0] / wrapper.box[1][1] < 3 / 2:
                ghost = zeros((img.shape[0], img.shape[1]), dtype=uint8)  # todo optimize if it lives on
                box = cv2.boxPoints(wrapper.box, points=zeros((4, 2), dtype=uint8))
                cv2.fillConvexPoly(ghost, box.astype(int32), color=(1, 1, 1))
                ghost *= fg / 255
                # cv2.drawContours(colors, contours8, wrapper.pos, color=(255, 255, 255), thickness=-1)

                # the foreground pixels inside the bounding box must be at least 80% non zero
                if 0.8 < npsum(ghost) / wrapper.area:
                    c = int(sum([pt[0] for pt in box]) / len(box)), int(sum([pt[1] for pt in box]) / len(box))
                    cv2.circle(img, c, int(expected_radius), (0, 0, 255), thickness=2)
                    # cv2.fillConvexPoly(roi_mask, box.astype(int32), color=(1, 1, 1))
                    diff = absolute(img.astype(float32) - self.background) * ghost[:, :, None].astype(float32)
                    if not 0 <= npmin(diff):
                        raise AssertionError("Expected 0 <= diff but min value is %d" % npmin(diff))
                    if not npmax(diff) <= 255:
                        raise AssertionError("Expected diff <= 255 but max value is %d" % npmax(diff))
                    sign = 1 if npsum(self.background*ghost[:, :, None]) < npsum(img*ghost[:, :, None]) else -1
                    meand = sign * npsum(diff) / wrapper.area
                    if 100 < abs(meand):
                        x, y = self._posgrid.get_intersection(c)
                        if self.is_empty(x, y):
                            inter = self.get_intersection(x, y)
                            inter.cleanup(self.total_f_processed - 15)
                            move = (B if meand < 0 else W, x, y)
                            inter.append(self.total_f_processed, move)
                            if inter.is_positive():
                                self.suggest(Move("cv", ctuple=move))

        self.metadata.insert(0, "frames : {0}".format(self.total_f_processed))
        # self.metadata.append("len(candidates): %d" % len(self.candidates))
        # if self.lastpos is not None:
        #     cv2.circle(img, self.lastpos, int(expected_radius), (0, 0, 255))
        self._show(img, loc=(1200, 600))

    @staticmethod
    def extract_contours(fg, expected_radius):
        """
        Extracts contours from the foreground mask that could correspond to a stone.
        Contours are sorted by enclosing circle area ascending.

        """
        ret, fg_noshad = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)  # discard the shadows (in grey)
        # try to remove some pollution to keep nice blobs
        smoothed = fg_noshad.copy()
        passes = 3
        for i in range(passes):
            cv2.erode(smoothed, (5, 5), dst=smoothed)
        prepared = cv2.Canny(smoothed, 25, 75)
        _, contours, hierarchy = cv2.findContours(prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (4/3 * expected_radius) ** 2
        max_area = (3 * expected_radius) ** 2
        sorted_conts = sort_contours_box(contours, area_bounds=(min_area, max_area))
        return sorted_conts, contours

    def _window_name(self):
        return BackgroundSub2.label


class Intersection():

    def __init__(self):
        self.history = []  # the results of foreground analysis for this intersection over time
        self.positive_threshold = 5

    def append(self, frame, move):
        assert not self.is_positive(), "intersection should already have been marked as occupied."
        # todo what if we find the same intersection more than once in the same frame ? consider it pollution ?
        self.history.append((frame, move))

    def is_positive(self):
        # todo maybe add some kind of check on the positions ? if so the real coordinates have to be stored as well
        return self.positive_threshold <= len(self.history)

    def cleanup(self, oldest_f):
        """
        Delete any move that has occurred before frame 'oldest_f'.

        """
        i = 0
        while i < len(self.history) and self.history[i][0] < oldest_f:
            i += 1
        self.history = self.history[i:]  # assume the list is sorted, and keep only the last up-to-date part

    def clear(self):
        self.history.clear()




















