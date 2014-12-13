
from numpy import uint8, zeros, ndarray, zeros_like, sum as npsum, mean as npmean
from numpy.ma import maximum
import cv2

from camkifu.core.imgutil import draw_contours_multicolor
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W, E

__author__ = 'Arnaud Peloquin'


class SfContours(StonesFinder):
    """
    Stones finder based on contours analysis.

    """

    label = "SF-Contours"

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.accu = zeros((self._posgrid.size, self._posgrid.size, 3), dtype=uint8)

    def _find(self, goban_img: ndarray):
        canvas = zeros((self._posgrid.size, self._posgrid.size, 3), dtype=uint8)
        # stones = self.find_stones(goban_img, c_start=6, c_end=13, canvas=canvas)
        stones = self.find_stones(goban_img, canvas=canvas)
        if stones is not None:
            temp = self.draw_stones(stones)
            self._show(maximum(canvas, temp))

    def _learn(self):
        pass

    def find_stones(self, img:  ndarray, r_start=0, r_end=gsize, c_start=0, c_end=gsize, canvas: ndarray=None):
        x0, y0, _, _ = self.getrect(r_start, c_start)
        _, _, x1, y1 = self.getrect(r_end - 1, c_end - 1)
        subimg = img[x0:x1, y0:y1]
        canny = self.get_canny(subimg)
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # todo the whole first part could be done on a larger zone than the desired subregion (to have more comp data)
        mask = zeros_like(canny)
        for cont in self._filter_contours(contours):
            # todo if solution retained, mark nearby intersections for later analysis (so that others can be ignored)
            cv2.drawContours(mask, [cv2.convexHull(cont)], 0, 1, thickness=-1)
        subgray = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)  # todo diff per color ?
        masked_gray = subgray * mask

        # zones:
        #       ¤ first channel (zone[:,:,0]    indicate whether or not each zone is masked
        #       ¤ second channel (zone[:,:,1]   store the mean pixel value of each zone
        zones = zeros((r_end - r_start, c_end - c_start, 2), dtype=uint8)
        for r in range(zones.shape[0]):
            for c in range(zones.shape[1]):
                a0, b0, a1, b1 = self.getrect(r + r_start, c + c_start)
                area = (a1 - a0) * (b1 - b0)
                visible_area = npsum(mask[a0 - x0:a1 - x0, b0 - y0:b1 - y0])  # count the positive (=1) mask pixels
                # a zone is masked if more than 60% of its pixels are masked.
                if 0.4 * area < visible_area:
                    zones[r, c, 0] = 1  # not masked
                    stone_mean = npsum(masked_gray[a0 - x0:a1 - x0, b0 - y0:b1 - y0]) / visible_area
                    zones[r, c, 1] = stone_mean
                else:
                    zones[r, c, 0] = 0  # masked
                    # todo optimize: compute only when in need of comparison (and not at every location)
                    zones[r, c, 1] = npmean(subgray[a0 - x0:a1 - x0, b0 - y0:b1 - y0])
        stones = zeros((gsize, gsize), dtype=object)
        stones[:] = E
        for r in range(zones.shape[0]):
            for c in range(zones.shape[1]):
                if zones[r, c, 0]:
                    self.find_color(r, c, zones, stones[r_start:r_end, c_start:c_end])
        if canvas is not None:
            draw_contours_multicolor(canvas[x0:x1, y0:y1], list(self._filter_contours(contours)))
        return stones

    def find_color(self, r, c, zones: ndarray, stones: ndarray):
        """
        Compare the (r, c) intersection's zone with its neighbours to determine whether it's a stone or not,
        and of which color.

        The results are aggregated in the 'stones' 2D array (supposed to be a sub-array only of the whole goban).

        """
        colors = set()
        added = 0
        for i in range(-1, 2):
            if 0 <= r + i < zones.shape[0]:
                for j in range(-1, 2):
                    if 0 == i and 0 == j:
                        continue
                    if 0 <= c + j < zones.shape[1]:
                        neigh = zones[r + i, c + j]
                        diff = int(zones[r, c, 1]) - int(neigh[1])  # convert to int to allow negative values
                        min_val = min(zones[r, c, 1], neigh[1])
                        # comparison with (supposedly) empty neighbour
                        if not neigh[0]:
                            # at least 40 points difference (in greyscale intensity for now)
                            if 40 < abs(diff):  # todo make that more dynamic
                                colors.add(B if diff < 0 else W)
                                added += 1
                        # comparison with (supposedly) stone neighbour
                        else:
                            # can only compare to already found stones, unless a more elaborated structure is created
                            if i < 1 and j < 1:
                                neigh_stone = stones[r + i, c + j]
                                if neigh_stone not in (B, W):
                                    continue
                                # less than 10% difference relatively to smallest val : ally stone
                                if abs(diff) < min_val * 0.1:
                                    colors.add(neigh_stone)
                                    added += 1
                                # at least 100% difference relatively to smallest val : enemy stone
                                elif min_val < abs(diff):
                                    colors.add(B if neigh_stone is W else (W if neigh_stone is B else E))
                                    added += 1
                        if added == 3: break
            if added == 3:
                if len(colors) == 1:
                    stones[r, c] = colors.pop()
                break

    def _filter_contours(self, contours):
        """
        Yield the subset of the provided contours that respect a bunch of constraints.

        """
        radius = self.stone_radius()
        for cont in contours:
            # it takes a minimum amount of points to describe the contour of a stone
            if cont.shape[0] < 10:
                continue
            box = cv2.minAreaRect(cont)
            # ignore contours that are too big, since in that case a good kmeans would probably do a more robust job.
            if 10 * radius < box[1][0] or 10 * radius < box[1][1]:
                continue
            yield cont

    def get_canny(self, img):
        median = cv2.medianBlur(img, 13)
        median = cv2.medianBlur(median, 7)  # todo play with median size / iterations a bit
        grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        otsu, _ = cv2.threshold(grey, 12, 255, cv2.THRESH_OTSU)
        return cv2.Canny(median, otsu / 2, otsu)

    def _window_name(self):
        return SfContours.label
