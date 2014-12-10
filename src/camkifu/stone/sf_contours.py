from math import sin, cos, radians

from numpy import uint8, zeros, ndarray, empty, mean as npmean
import cv2

from camkifu.core.imgutil import norm, draw_contours_multicolor
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
        rs, re = 0, 19
        cs, ce = 6, 13
        stones = self.find_stones(goban_img, r_start=rs, r_end=re, c_start=cs, c_end=ce)
        self.display_stones(stones)

    def _learn(self):
        pass

    def find_stones(self, img: ndarray, r_start=0, r_end=gsize, c_start=0, c_end=gsize, canvas: ndarray=None):
        x0, y0, _, _ = self.getrect(r_start, c_start)
        _, _, x1, y1 = self.getrect(r_end - 1, c_end - 1)
        median = img[x0:x1, y0:y1].copy()
        median = cv2.medianBlur(median, 13)
        median = cv2.medianBlur(median, 7)  # todo play with median size / iterations a bit
        grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        otsu, _ = cv2.threshold(grey, 12, 255, cv2.THRESH_OTSU)
        canny = cv2.Canny(median, otsu / 2, otsu)
        # canny = cv2.Canny(goban_img, 25, 75)
        centers, contours = self._analyse_contours(canny)
        stones = self.find_colors(img, [(x + y0, y + x0) for x, y in centers])  # in opencv coordinates system
        if canvas is not None:
            canvas[x0:x1, y0:y1] /= 4
            draw_contours_multicolor(canvas[x0:x1, y0:y1], contours)
        return stones

    def _analyse_contours(self, img: ndarray, row_start=0, row_end=gsize, col_start=0, col_end=gsize):
        """
        Return a list of points indicating likely locations of stones. Based on contours analysis in "img".
        The search can be confined to a sub-region of the image using arguments row_start, row_end, col_start, col_end.

        """
        x0, y0, _, _ = self.getrect(row_start, col_start)
        _, _, x1, y1 = self.getrect(row_end - 1, col_end - 1)
        subregion = img[x0:x1, y0:y1]
        ghost = zeros((x1 - x0, y1 - y0), dtype=uint8)
        # todo experiment with something else than retr_external ?
        _, contours, hierarchy = cv2.findContours(subregion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []  # one maximum per zone, in opencv coordinates system
        # todo seeing that contours are often shattered into pieces, filtering may be counter-productive when
        # used with distance transform -> experiment without.
        for cont in self._filter_contours(contours):
            # compute the distance matrix from contour: the higher the value, the further the point from the contour
            cv2.drawContours(ghost, [cont], 0, (255, 0, 255))
            ry0, rx0, dy, dx = cv2.boundingRect(cont)  # contour region (not rotated rectangle)
            negative = empty((dx, dy), dtype=uint8)
            negative[:] = 255
            negative -= ghost[rx0:rx0 + dx, ry0:ry0 + dy]
            distance_mtx = cv2.distanceTransform(negative, cv2.DIST_L2, cv2.DIST_MASK_5)
            for a, b in self._find_centers(distance_mtx):
                centers.append((a + y0 + ry0, b + x0 + rx0))  # put back all offsets to integrate into global img
        return centers, contours

    def _filter_contours(self, contours):
        """
        Yield the subset of the provided contours that respect a bunch of constraints. The objective is to discard
        non-stone contours, even though some good contours may also be lost on the way.

        """
        radius = self.stone_radius()
        for cont in contours:
            # it takes a minimum amount of points to describe the contour of a stone
            if cont.shape[0] < 10:
                continue
            box = cv2.minAreaRect(cont)
            # ignore contours that have at least one side too small
            if min(box[1]) < 3 / 2 * radius:
                continue
            # ignore contours that are too big, since in that case a good kmeans would probably do a more robust job.
            if 8 * radius < box[1][0] or 8 * radius < box[1][1]:
                continue
            # ignore badly orientated bounding rect, for big contours only
            angle = radians(box[2])
            if 2.5 * radius < max(box[1]) and max(abs(cos(angle)), abs(sin(angle))) < 0.998:
                continue
            yield cont

    def _find_centers(self, distance_mtx):
        """
        Yield image points that are likely to represent the center of a stone, based on the following:

        Locate the most "distant" point in each (estimated) stone's zone. From these candidates, only keep
        those located in the zone's center (ignore candidates hitting the wall). The objective is to retain
        points that are best surrounded by contour curves, and are thus likely to indicate a stone's center.

        """
        radius = self.stone_radius()
        dx, dy = distance_mtx.shape[0], distance_mtx.shape[1]
        nb_rows = int(round(dx / 2 / radius))
        row_width = int(dx / nb_rows)
        nb_cols = int(round(dy / 2 / radius))
        col_width = int(dy / nb_cols)
        for row in range(nb_rows):
            rs, re = row * row_width, (row + 1) * row_width  # row start, row end
            for col in range(nb_cols):
                cs, ce = col * col_width, (col + 1) * col_width  # col start, col end
                _, _, _, maxloc = cv2.minMaxLoc(distance_mtx[rs:re + 1, cs:ce + 1])
                if min(row_width, col_width) / 4 < norm(maxloc, ((ce - cs)/2, (re - rs)/2)):
                    continue
                yield cs + maxloc[0], rs + maxloc[1]

    def find_colors(self, img, centers) -> ndarray:
        """
        Get for each center the closest goban intersection. Compare this intersection with its neighbours to
        determine the color of a potential stone there.

        There is a 10% difference entry ticket : no color is set if the intersection is less than 10% different
        from its empty neighbours.

        """
        # note : can't apply mask here, since we compare different intersections, which may not have the same number
        # of masked pixels (zones may not be of the same size depending on grid adjustment implementations.
        stones = zeros((gsize, gsize), dtype=object)
        stones[:] = E
        for a, b in centers:
            x, y = self._posgrid.closest_intersection((b, a))
            x0, y0, x1, y1 = self.getrect(x, y)
            stone_mean = npmean(img[x0:x1, y0:y1])  # greyscale will be used in the comparison
            neighbs, count = 0, 0  # neighbours means , count of neighbours
            for i in range(-1, 2):
                if not 0 <= x + i < gsize:
                    continue
                for j in range(-1, 2):
                    if not 0 <= y + j < gsize:
                        continue
                    if not self.is_empty(x + i, y + j):
                        continue
                    x0, y0, x1, y1 = self.getrect(x + i, y + j)
                    neighbs += npmean(img[x0:x1, y0:y1])
                    count += 1
                    if 2 < count:
                        break  # just get a feeling, no need to accumulate the potential 8 neighbours
            if 0 < count:  # todo implemement non-empty neighbours comparison
                if stone_mean < neighbs / count * 0.9:
                    stones[x][y] = B
                elif neighbs / count * 1.1 < stone_mean:
                    stones[x][y] = W
        return stones
    # def _draw_metadata(self, img, latency, thread):
    # pass

    def _window_name(self):
        return SfContours.label
