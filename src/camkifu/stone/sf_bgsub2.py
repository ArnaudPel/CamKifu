from math import pi
from queue import Empty

import cv2
from numpy import zeros_like, zeros, uint8, int32, empty, empty_like, sum as npsum
from time import time
import sys

from camkifu.core.imgutil import draw_str, sort_contours_circle, draw_contours_multicolor, connect_clusters, \
    sort_contours_box
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
        self.fg_mask = None
        self.candidates = []  # candidates for the next added stone
        self.candid_acc = 0  # the number of frames since last candidate list clear
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
        Return True when enough images have been applied to the background model.

        """
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

    def search(self, img):
        """
        Try to detect stones by comparing against (cached) background colors.

        """
        expected_radius = max(*img.shape) / gsize / 2  # the approximation of the radius of a stone, in pixels

        # todo read paper about MOG2, in order to know how to use it properly here
        learn = 0 if self.total_f_processed % 5 else 0.01
        fg = self._bg_model.apply(img, fgmask=self.fg_mask, learningRate=learn)
        sorted_conts, contours = self.extract_contours(fg, expected_radius)

        # search for a contour that could be a new stone
        # colors = zeros_like(img)
        # draw_contours_multicolor(colors, contours)

        for wrapper in sorted_conts:
            if 2 / 3 < wrapper.box[1][0] / wrapper.box[1][1] < 3 / 2:
                ghost = zeros((img.shape[0], img.shape[1]), dtype=uint8)
                box = cv2.boxPoints(wrapper.box, points=zeros((4, 2), dtype=uint8))
                cv2.fillConvexPoly(ghost, box.astype(int32), color=(1, 1, 1))
                ghost *= fg / 255
                percent = npsum(ghost) / wrapper.area * 100
                # cv2.drawContours(colors, contours, wrapper.pos, color=(255, 255, 255), thickness=-1)

                # finally display contours accepted as candidates
                if 80 < percent:
                    c = int(sum([pt[0] for pt in box]) / len(box)), int(sum([pt[1] for pt in box]) / len(box))
                    cv2.circle(img, c, int(expected_radius), (0, 0, 255), thickness=2)

        self.metadata.insert(0, "frames : {0}".format(self.total_f_processed))
        # self.metadata.append("len(candidates): %d" % len(self.candidates))
        if self.lastpos is not None:
            cv2.circle(img, self.lastpos, int(expected_radius), (0, 0, 255))
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
























