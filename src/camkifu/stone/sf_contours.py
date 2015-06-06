import itertools
import math

import numpy as np
import cv2

from camkifu.core import imgutil
import camkifu.stone
from golib.config.golib_conf import gsize, B, W, E


class SfContours(camkifu.stone.StonesFinder):
    """ Stones finder implementation based on contours analysis.

    Two kinds of contours are analysed:
        ¤ On current image, to try to find any stone.
        ¤ On foreground mask, to try to find a newly put stone.

    This class has the ability to be used as a "standalone" finder to have a look, but as it is not designed to
    work in high stones-density context, it is put to better use when combined with other approaches. See SfMeta.
    """

    def __init__(self, vmanager):
        super().__init__(vmanager)

    def _find(self, goban_img: np.ndarray):
        """ Wait for background initialization, then run stones detection and display results without submitting them.

        Note: SfMeta is using this class abilities, but it does via find_stones (thus bypassing this method).

        Args:
            goban_img: ndarray
                The Goban image, in the canonical frame.
        """
        if self.bg_init_frames < self.total_f_processed:
            canvas = np.zeros((self._posgrid.size, self._posgrid.size, 3), dtype=np.uint8)
            # stones = self.find_stones(goban_img, cs=6, ce=13, canvas=canvas)
            stones = self.find_stones(goban_img, canvas=canvas)
            if stones is not None:
                temp = self.draw_stones(stones)
                self._show(np.maximum(canvas, temp))
        else:
            self.display_bg_sampling(goban_img.shape)

    def _learn(self):
        pass

    def find_stones(self, img:  np.ndarray, rs=0, re=gsize, cs=0, ce=gsize, canvas: np.ndarray=None):
        """ The stones detection main algorithm, which is based on contours analysis.

        Two kinds of contours are extracted:
            ¤ On current image, to try to find any stone.
            ¤ On foreground mask, to try to find a newly put stone.

        A mask is created by drawing the convex hull of each contour on a black image. After having applied this
        mask on the original image, each intersection pixels are summed and analysed: if there are enough non-zero,
        a color is determined for that intersection based on neighbour intersections comparison.

        Args:
            img: ndarray
                The Goban image.
            canvas: ndarray
                An optional image on which to draw some results.
                        rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.

        Returns stones: ndarray
            The stones found, a 2D-matrix containing either B, W or E.
        """
        x0, y0, _, _ = self.getrect(rs, cs)
        _, _, x1, y1 = self.getrect(re - 1, ce - 1)
        contours_fg = self.analyse_fg(x0, y0, x1, y1, canvas=canvas)
        subimg = img[x0:x1, y0:y1]
        canny = self.get_canny(subimg)
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(subimg)
        contours_img = list(self._filter_contours(contours))
        for cont in itertools.chain(contours_fg, contours_img):
            cv2.drawContours(mask, [cv2.convexHull(cont)], 0, (1, 1, 1), thickness=-1)
        visible_sub = subimg * mask
        masked_sub = subimg * (1 - mask)

        # zones:
        #       ¤ first channel  (zone[:,:,0])     indicate whether or not each zone is masked
        #       ¤ other channels (zone[:,:,1:4])   store the mean pixel value of each zone, for each color (B, G, R)
        zones = np.zeros((re - rs, ce - cs, 4), dtype=np.int16)
        for r in range(zones.shape[0]):
            for c in range(zones.shape[1]):
                a0, b0, a1, b1 = self.getrect(r + rs, c + cs)
                area = (a1 - a0) * (b1 - b0)
                visible_area = np.sum(mask[a0 - x0:a1 - x0, b0 - y0:b1 - y0, 0])  # count the visible (=1) mask pixels
                # a zone is masked if more than 60% of its pixels are masked.
                if 0.4 * area < visible_area:
                    zones[r, c, 0] = 1  # not masked
                    self._norm_channels(zones[r, c], visible_sub[a0 - x0:a1 - x0, b0 - y0:b1 - y0], visible_area)
                else:
                    zones[r, c, 0] = 0  # masked
                    self._norm_channels(zones[r, c], masked_sub[a0 - x0:a1 - x0, b0 - y0:b1 - y0], area - visible_area)
        stones = np.zeros((gsize, gsize), dtype=object)
        stones[:] = E
        for r in range(zones.shape[0]):
            for c in range(zones.shape[1]):
                if zones[r, c, 0]:
                    self.find_color(r, c, zones, stones[rs:re, cs:ce])
        if canvas is not None:
            imgutil.draw_contours_multicolor(canvas[x0:x1, y0:y1], contours_img)
        return stones

    @staticmethod
    def _norm_channels(slot, img, norm):
        """ Compute for each color channel of img a normalized value (B, G, and R) and store it in slot.

        Args:
            slot: array
                Where to store one value for each color channel (expected to be of length 4).
            img: ndarray
                The image to analyse.
            norm: int
                The normalization factor.
        """
        for k in range(3):
            slot[k + 1] = np.sum(img[:, :, k]) / norm

    @staticmethod
    def find_color(r, c, zones: np.ndarray, stones: np.ndarray):
        """ Compare the (r, c) intersection's zone with its neighbours to determine its color (B, W, or E).

        Note: 'stones' is used both as data and result slot.

        Args:
            r: int
            c: int
                The row and column of the intersection to analyse.
            zones: ndarray
                Data of all the zones (intersections) of the Goban.
            stones: ndarray
                Where to aggregate the results, at the (r, c) position. Is also used to get the already found stones.
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
                        raw_diff = zones[r, c, 1:4] - neigh[1:4]
                        sign = -1 if np.sum(raw_diff) < 0 else 1
                        diff = sign * np.sum(np.absolute(raw_diff))
                        # comparison with (supposedly) empty neighbour
                        if not neigh[0]:
                            # at least 140 points absolute difference between the channels
                            if 100 < abs(diff):
                                colors.add(B if diff < 0 else W)
                                added += 1
                            elif abs(diff) < 70:
                                colors.add(E)
                                added = 3  # break whole search
                        # comparison with (supposedly) stone neighbour
                        else:
                            min_val = min(np.sum(zones[r, c, 1:4]), np.sum(neigh[1:4]))
                            # can only compare to already found stones, unless a more elaborated structure is created
                            if i < 1 and j < 1:
                                neigh_stone = stones[r + i, c + j]
                                if neigh_stone not in (B, W):
                                    continue
                                # less than 10% difference relatively to smallest val : ally stone
                                if abs(diff) < min_val * 0.1:
                                    colors.add(neigh_stone)
                                    added += 1
                                # at least 100% difference relatively to smallest val : enemy stone
                                elif min_val < abs(diff):
                                    colors.add(B if neigh_stone is W else (W if neigh_stone is B else E))
                                    added += 1
                        if added == 3: break
            if added == 3:
                if len(colors) == 1:
                    stones[r, c] = colors.pop()
                break

    def _filter_contours(self, contours):
        """ Filter the provided contours to respect some constraints. Basically ignore those too small or too big.

        Args:
            contours: iterable
                The contours to filter.

        Yields cont: contour
            The next contour respecting the constraints.
        """
        radius = self.stone_radius()
        for cont in contours:
            # it takes a minimum amount of points to describe the contour of a stone
            if cont.shape[0] < 10:
                continue
            box = cv2.minAreaRect(cont)
            # ignore contours that are too big, since in that case a good kmeans would probably do a more robust job.
            if 10 * radius < max(box[1]):
                continue
            yield cont

    def analyse_fg(self, x0, y0, x1, y1, canvas: np.ndarray=None) -> list:
        """ Try to get stone(s) contour(s) from foreground analysis, to help better detect newly played stones.

        In order to reduce false positives, contours found are filtered with a bunch of constraints:
            ¤ size:   see extract_contours_fg()
            ¤ shape:  see below, the use of _find_centers()

        Args:
            x0: int
            y0: int
            x1: int
            y1: int
                Delimiters of the foreground subregion where contours should be searched.
            canvas: ndarray
                An optional image where to draw some search results.

        Return filtered: list
            The interesting contours that could be found in the foreground mask.
        """
        sub_fg = self.get_foreground()[x0:x1, y0:y1]
        sub_canvas = None
        if canvas is not None:
            sub_canvas = canvas[x0:x1, y0:y1]
        contours = self.extract_contours_fg(sub_fg, canvas=sub_canvas)
        filtered = []
        ghost = np.zeros(sub_fg.shape, dtype=np.uint8)
        for cont in contours:
            # compute the distance matrix for each pixel: the higher the value, the further the pixel from the contour
            cv2.drawContours(ghost, [cont], 0, (255, 0, 255))
            ry0, rx0, dy, dx = cv2.boundingRect(cont)  # contour region (not rotated rectangle)
            negative = np.empty((dx, dy), dtype=np.uint8)
            negative[:] = 255
            negative -= ghost[rx0:rx0 + dx, ry0:ry0 + dy]
            distance_mtx = cv2.distanceTransform(negative, cv2.DIST_L2, cv2.DIST_MASK_5)
            for a, b in self._find_centers(distance_mtx):
                # if at least one center found, keep contour, since it means its shape is "closed" enough
                # ignore the center itself, unless in need to draw it
                filtered.append(cont)
                if canvas is not None:
                    center = (a + y0 + ry0, b + x0 + rx0)  # put back all offsets to integrate into global img
                    cv2.circle(canvas, center, 3, (0, 0, 255), thickness=-1)
                break
        return filtered

    def extract_contours_fg(self, sub_fg: np.ndarray, canvas: np.ndarray=None):
        """ Extract contours from the foreground mask that could correspond to a stone.

        Contours are sorted by enclosing circle area ascending.

        Args:
            sub_fg: ndarray
                The foreground subregion where to look for contours.
            canvas: ndarray
                An optional image where to draw the contours retained.

        Returns filtered: list
            The interesting contours that could be found in the foreground mask.
        """
        # try to remove some pollution to keep nice blobs
        smoothed = cv2.morphologyEx(sub_fg, cv2.MORPH_OPEN, (5, 5), iterations=3)
        canny = cv2.Canny(smoothed, 25, 75)
        _, contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        radius = self.stone_radius()
        filtered = []
        bigenough = []  # the contours that are big enough but have been discarded by subsequent constraints
        for cont in contours:
            # it takes a minimum amount of points to describe the contour of a stone
            if cont.shape[0] < 10:
                continue
            box = cv2.minAreaRect(cont)
            # ignore contours that have at least one side too small
            if min(box[1]) < 3 / 2 * radius:
                continue
            bigenough.append(cont)
            # ignore contours that are too big (two stones max)
            if 5 * radius < max(box[1]):
                continue
            # ignore badly orientated bounding rect, for big contours only
            angle = math.radians(box[2])
            if 2.5 * radius < max(box[1]) and max(abs(math.cos(angle)), abs(math.sin(angle))) < 0.97:
                continue
            # ignore contours which interior is too black
            hull = cv2.convexHull(cont)
            y0, x0, dy, dx = cv2.boundingRect(hull)
            ghost = np.zeros((dx, dy), dtype=np.uint8)
            cv2.drawContours(ghost, [hull], 0, color=1, thickness=-1, offset=(-y0, -x0))
            ratio = np.sum(sub_fg[x0:x0 + dx, y0:y0 + dy] * (ghost == 1)) / dx / dy / 255
            if ratio < 0.3:
                continue
            filtered.append(cont)
        if canvas is not None:
            cv2.drawContours(canvas, bigenough, -1, (0, 0, 255))
            cv2.drawContours(canvas, filtered, -1, (0, 255, 0))
        return filtered

    def _find_centers(self, distance_mtx):
        """ Find image points that are likely to represent the center of a stone.

        Locate the most "distant" point from the contour in each (estimated) stone's zone. From these candidates,
        only keep those located in the zone's center (ignore candidates hitting the wall). The objective is to
        retain points that are best surrounded by contour curves, and are thus likely to indicate a stone's center.

        Args:
            distance_mtx: as returned by cv2.distanceTransform()
                The distance of image each pixel to the closest contour pixel.

        Yields x, y: int, int
            The next valid center.
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
                # discard if ==> smallest side / 3 < distance(maxloc, center)
                if min(row_width, col_width) / 3 < imgutil.norm(maxloc, ((ce - cs) / 2, (re - rs) / 2)):
                    continue
                yield cs + maxloc[0], rs + maxloc[1]

    @staticmethod
    def get_canny(img):
        """ Smooth image using median blur, then call Canny with Otsu thresholds.
        """
        median = cv2.medianBlur(img, 13)
        median = cv2.medianBlur(median, 7)
        grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        otsu, _ = cv2.threshold(grey, 12, 255, cv2.THRESH_OTSU)
        return cv2.Canny(median, otsu / 2, otsu)

    def _window_name(self):
        return SfContours.__name__
