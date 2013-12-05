import cv2
from numpy import zeros_like, mean, zeros, uint8, uint16, int8, int16, empty, sum as npsum
from numpy.ma import minimum, maximum, absolute
from core.imgutil import draw_str

from stone.stonesbase import StonesFinder
from golib_conf import gsize, player_color

__author__ = 'Kohistan'

nb_bg_sample = 3  # todo move that to config or inline


class BackgroundSub2(StonesFinder):
    """
    Save background data using sample(img)
    Perform background subtraction operations in order to detect stones.

    self.stones -- the matrix of stones found so far (0:None, 1:Black, 2:White)
    """

    def __init__(self, vmanager, rectifier):
        super(BackgroundSub2, self).__init__(vmanager, rectifier)
        self.bindings['s'] = self.reset

        self._background = None
        self.dosample = nb_bg_sample
        self.lastpos = None

    def _find(self, goban):
        goban_img = cv2.medianBlur(goban, 5)
        if self.dosample:
            self.sample(goban_img)
        else:
            self.detect(goban_img)
        self._drawgrid(goban_img)
        self._show(goban_img, name="Goban frame")

    def reset(self):
        super(BackgroundSub2, self).reset()
        self._background = None
        self.lastpos = None
        self.dosample = nb_bg_sample

    def sample(self, img):
        if self._background is None:
            self._background = zeros_like(img, dtype=uint8)
        self._background += img
        self.dosample -= 1
        if not self.dosample:
            # store mean background
            self._background /= nb_bg_sample
        draw_str(img, (40, 60), "Background sampled {0}/{1}".format(nb_bg_sample - self.dosample, nb_bg_sample))

    def detect(self, img):
        """
        Try to detect stones by comparing against neighbour colors.

        mask -- a matrix of shape (gsize, gsize) providing positions of already known stones, as follow.
                0: empty position
                1: white position
                2: black position

        """
        pos = None
        val = 0

        # todo surround with try/except and reset if img size has changed
        imsub = cv2.subtract(img, self._background, dtype=cv2.CV_16S)

        meanall = mean(imsub)
        draw_str(img, (40, 80), "Mean : %.1f" % meanall)

        absmean = mean(absolute(imsub))
        draw_str(img, (40, 100), "AMean: %.1f" % absmean)

        #diffs = zeros((gsize, gsize, 3), int16)
        #for x in range(gsize):
        #    for y in range(gsize):
        #        zones, points = self._getzones(imsub, x, y)
        #        factor = 0
        #        for zone in zones:
        #            diffs[x][y][0] += mean(zone[:, :, 0])
        #            diffs[x][y][1] += mean(zone[:, :, 1])
        #            diffs[x][y][2] += mean(zone[:, :, 2])
        #            factor += 1
        #        diffs[x][y] /= factor  # get the actual mean for each color

        # for debug display
        #devim = zeros((gsize * 30, gsize * 30, 3), dtype=uint8)
        #for x in range(gsize):
        #    for y in range(gsize):
        #        r = abs(int(diffs[x][y][0]))
        #        g = abs(int(diffs[x][y][1]))
        #        b = abs(int(diffs[x][y][2]))
        #        cv2.rectangle(devim, (x * gsize, y * gsize), ((x + 1) * gsize, (y + 1) * gsize), (r, g, b),
        #                          thickness=-1)
        #self._show(devim, "Diff Zones")

        # validate stone when it has been found twice, anti-false-positive measure
        if pos is not None:
            if self.lastpos == pos:
                self.stones[pos] = val
                row = chr(97 + pos[1])
                col = chr(97 + pos[0])
                print "{0}[{1}{2}]".format(player_color[val], row, col)
                self.vmanager.controller.pipe("append", (player_color[val], row, col))
            else:
                self.lastpos = pos





























