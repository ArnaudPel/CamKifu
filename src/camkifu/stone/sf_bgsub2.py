from math import pi
from queue import Empty

import cv2
from numpy import zeros_like, zeros, uint8, int32, empty, empty_like
from time import time

from camkifu.core.imgutil import draw_str, sort_contours_circle, draw_contours_multicolor
from golib.model.move import Move
from camkifu.stone.stonesfinder import StonesFinder, compare, evalz
from golib.config.golib_conf import gsize, B, W, E


__author__ = 'Arnaud Peloquin'

# the number of background sampling frames before trying to detect stones.
bg_init_number = 25

# possible states for a BackgroundSub instance :
sampling = "sampling"
watching = "watching"
searching = "searching"

# max number of 'untouched' searches before triggering the 'watching' state
untouched_threshold = 7


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
        if self._bg_initialization < bg_init_number:
            black = zeros((img.shape[0], img.shape[1]), dtype=uint8)
            draw_str(black, int(black.shape[0] / 2 - 70), int(black.shape[1] / 2),
                     "SAMPLING ({0}/{1})".format(self._bg_initialization, bg_init_number))
            self._show(black)
            return False
        return True

    def search(self, img):
        """
        Try to detect stones by comparing against (cached) background colors.

        """
        # todo read paper about MOG2, in order to know how to use it properly here
        if not self.total_f_processed % 4:
            pos = None
            color = E
            fg = self._bg_model.apply(img, fgmask=self.fg_mask, learningRate=0.01)
            ret, fg = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
            self.metadata.insert(0, "state : " + self.state)
            self.metadata.append("frames : %d" % self.total_f_processed)

            # try to remove some pollution to keep nice blobs
            smoothed = fg.copy()
            passes = 3
            for i in range(passes):
                cv2.erode(smoothed, (5, 5), dst=smoothed)
            # for i in range(max(int(passes/2), 1)):
            #     cv2.dilate(smoothed, (5, 5), dst=smoothed)
            # smoothed = cv2.morphologyEx(fg, cv2.MORPH_OPEN, (5, 5))
            # smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, (5, 5))
            # smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, (5, 5))

            prepared = cv2.Canny(smoothed, 25, 75)
            _, contours, hierarchy = cv2.findContours(prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            expected_radius = max(*img.shape) / (2*gsize)
            min_area = pi * (expected_radius/1.5)**2
            max_area = pi * (expected_radius*1.5)**2
            # sorted_conts = sort_contours_circle(contours)
            sorted_conts = sort_contours_circle(contours, area_bounds=(min_area, max_area))
            # colors = zeros_like(img)
            # draw_contours_multicolor(colors, contours)
            for wrapper in sorted_conts:
                c = int(wrapper.circle[0][0]), int(wrapper.circle[0][1])
                cv2.circle(img, c, int(wrapper.circle[1]), (0, 0, 255), thickness=2)

            # self._show(fg, name="Foreground", loc=(800, 600))
            # self._show(prepared, name="SF-Canny", loc=(800, 600))
            self._show(img, loc=(1200, 600))
            if pos is not None:
                if self.lastpos == pos:
                    self.suggest(Move("cv", ctuple=(color, pos[0], pos[1])))
                    self.state = sampling
                else:
                    self.lastpos = pos
            else:
                self.nb_untouched += 1

    def _window_name(self):
        return BackgroundSub2.label
























