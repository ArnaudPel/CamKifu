from queue import Empty
from time import time

import cv2
from numpy import zeros, uint8, int32, float32, sum as npsum, array, max as npmax,\
    min as npmin, mean as npmean
from numpy.ma import absolute

from camkifu.core.imgutil import draw_str, sort_contours_box
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W


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
        self.triggered = []

        self.state = sampling
        self.nb_untouched = 0  # the number of successive searches that detected no motion at all
        self.last_on = time()  # instant when last active. to be used to detect long sleeps.

    def _find(self, goban_img):
        filtered = cv2.medianBlur(goban_img, 7)  # todo search what's best here given the new bg modeling
        if self.state == sampling:
            done = self.sample(filtered)
            if done:
                self.state = searching
        else:
            self.search(filtered)
            self.last_on = time()

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
            draw_str(black, "SAMPLING ({0}/{1})".format(self._bg_initialization, bg_learning_frames),
                     int(black.shape[0] / 2 - 70), int(black.shape[1] / 2))
            self._show(black)
            return False
        return True

    def get_intersection(self, x, y):
        inter = self.intersections[x][y]
        if inter == 0:
            inter = IntersectionTrigger(x, y)
            self.intersections[x][y] = inter
        return inter

    @staticmethod
    def debug_check_diff(diff):
        # todo remove after a while
        if not -255 <= npmin(diff):
            raise AssertionError("Expected -255 <= diff but min value is %d" % npmin(diff))
        if not npmax(diff) <= 255:
            raise AssertionError("Expected diff <= 255 but max value is %d" % npmax(diff))

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

        global_diff = (img.astype(float32) - self.background)

        #  STEP ONE: TRIGGERS
        # search for a contour that could be a new stone. the unlikely areas have been trimmed already.
        for wrapper in sorted_conts:
            # todo extract constraint-checking methods so that they can be re-ordered, and also just to clean the code
            # the bounding box must be a rough square
            if 2 / 3 < wrapper.box[1][0] / wrapper.box[1][1] < 3 / 2:
                box = cv2.boxPoints(wrapper.box, points=zeros((4, 2), dtype=uint8))
                # cv2.drawContours(colors, contours8, wrapper.pos, color=(255, 255, 255), thickness=-1)

                # get the center of the box to determine which intersection of the goban has been triggered
                x_mass = int(sum([pt[0] for pt in box]) / len(box))
                y_mass = int(sum([pt[1] for pt in box]) / len(box))
                box_center = x_mass, y_mass
                x, y = self._posgrid.get_intersection(box_center)  # the related intersection of the goban

                if self.is_empty(x, y):
                    box_mask = zeros((img.shape[0], img.shape[1]), dtype=uint8)  # todo optimize if it lives on
                    cv2.fillConvexPoly(box_mask, box.astype(int32), color=(1, 1, 1))
                    box_mask *= fg / 255
                    # the foreground pixels inside the bounding box must be at least 80% non-zero
                    if 0.8 < npsum(box_mask) / wrapper.area:
                        cv2.circle(img, box_center, int(expected_radius), (0, 0, 255), thickness=2)
                        box_mask3d = box_mask.astype(float32)[:, :, None]
                        diff = global_diff * box_mask3d
                        self.debug_check_diff(diff)
                        meand = get_meand(diff, wrapper.area)
                        if 100 < abs(meand):
                            inter = self.get_intersection(x, y)
                            inter.trigger(meand, self.total_f_processed, box_mask3d, wrapper.area)

        # STEP 2: FOLLOW-UP
        nb_active = 0
        for x in range(gsize):
            for y in range(gsize):
                inter = self.intersections[x][y]
                if inter != 0:
                    inter.submit(global_diff, self.total_f_processed)
                    if len(inter.history):   # todo remove debug
                        nb_active += 1
                    move = inter.is_positive()
                    if move is not None:
                        self.suggest(*move)
                        self.intersections[x][y] = 0  # todo find a cleaner way to park occupied intersections

        self.metadata["frames : {}"] = self.total_f_processed
        self.metadata["active intersections : {}"].append(nb_active)
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


def get_meand(diff, norm):
    sign = - 1 if npsum(diff) < 0 else 1
    return sign * npsum(absolute(diff)) / norm


class IntersectionTrigger():

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.mask = None  # the mask defining the image region of the last trigger
        self.norm = None  # the norm of the image region of the last trigger (eg. its area)
        self.history = []
        self.memorize = 7  # number of frames after the last trigger submissions are accepted

    def trigger(self, meand, frame_nr, mask, norm):
        """
        To be called when the foreground has identified this intersection as a likely stone.

        """
        self.mask = mask
        self.norm = norm
        self.history.append((frame_nr, meand, True))

    def submit(self, diff_img, frame_nr):
        """
        To be called after the foreground has identified this intersection as a likely stone. ('after' meaning the
        next frame and on).

        """
        self.forget(frame_nr)
        if len(self.history):
            last_entry = self.history[-1]
            if last_entry[0] != frame_nr:
                meand = get_meand(diff_img * self.mask, self.norm)
                self.history.append((frame_nr, meand, False))
            elif not last_entry[2]:  # if the last entry is not a trigger
                raise ValueError("Submitted same frame twice to IntersectionTrigger")

    def forget(self, frame_nr):
        if len(self.history):
            while self.history[0][0] < frame_nr - self.memorize:
                self.history.pop(0)  # remove entries that are too old
            i = 0
            while i < len(self.history) and not self.history[i][2]:
                # delete non-trigger entries at beginning of the list if any
                i += 1
            self.history = self.history[i:]  # assume the list is sorted asc, and keep only the last part

    def is_positive(self):
        # if the sample has been run to its full possible extent, check conditions
        if self.memorize <= len(self.history):
            trig = 0
            for entry in self.history:
                if entry[2]:
                    trig += 1
                    if 2 < trig:
                        meand = npmean([x[1] for x in self.history])
                        print("meand: {}".format(meand))  # todo remove debug
                        if meand < -100 or 120 < (meand):
                            color = B if meand < 0 else W
                            return color, self.x, self.y
                        break
        return None