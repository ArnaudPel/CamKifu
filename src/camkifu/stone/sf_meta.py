from numpy import uint8, float32, zeros, ndarray, unique
import cv2
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_contours import SfContours

from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W, E

__author__ = 'Arnaud Peloquin'


class SfMeta(StonesFinder):
    """
    Work in progress.

    A "meta" stones finder at the moment, since it aims to aggregate contributions from the different stones finder
    experimented so far, with the hope to become something actually useful.

    """

    label = "SF-Meta"

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.accu = zeros((self._posgrid.size, self._posgrid.size, 3), dtype=uint8)
        self.cluster = SfClustering(None)  # set to None for safety, put vmanager when needed
        self.contour = SfContours(vmanager)  # this one needs a vmanager already
        self.contour._show = self._show

        # todo switch to global kmeans when 4 - 5 regions out of the 9 are positive ? + at least one per row and col
        self.kmeans_positive = zeros((3, 3), dtype=bool)  # todo instead of bool, store an evolving score for each sf
        self.histo_len = 10  # todo integrating a background detector should help lowering the history (less false posit)
        self.contour_accu = zeros((gsize, gsize, self.histo_len), dtype=object)
        self.contour_accu[:] = E

    def _find(self, goban_img: ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> assess clustering result with contours results, and complete with the former if validated

        """
        cont_stones = self.contour.find_stones(goban_img)
        clust_stones = zeros((gsize, gsize), dtype=object)
        grid = self.search_intersections(goban_img)
        img = goban_img.astype(float32)
        skipped_kmeans = 0  # todo remove debug
        split = 3
        i = 0
        for rs, re, cs, ce in self.subregions(split=split):
            contours = 0  # total number of stones found by sf_contours, to normalize
            lines = 0  # in-front count of lines to avoid running uncheckable clustering
            for r in range(rs, re):
                for c in range(cs, ce):
                    stone = cont_stones[r][c]
                    if stone in (B, W):
                        contours += 1
                        self.contour_accu[r][c][self.total_f_processed % self.histo_len] = stone  # could be outside
                    if sum(grid[r][c]) < 0:
                        lines += 1
            if lines < 4 and contours < 3:
                # no way to check clustering, ignore that zone
                # bad news of course
                skipped_kmeans += 1
                continue

            ratios, centers = self.cluster.cluster_colors(img, r_start=rs, r_end=re, c_start=cs, c_end=ce)

            lines, line_confs, result = self.cluster.check_line_conflicts(
                grid, ratios, centers, r_start=rs, r_end=re, c_start=cs, c_end=ce)

            if 3 < lines and 0.9 < line_confs / lines:
                cont_confs = 0  # confirmation of clustering results, by contours results
                for r in range(rs, re):
                    for c in range(cs, ce):
                        if cont_stones[r][c] in (B, W) and result[r][c] is cont_stones[r][c]:
                            cont_confs += 1
                if 2 < contours and 0.9 < cont_confs / contours:
                    clust_stones[rs:re, cs:ce] = result[rs:re, cs:ce]
                    self.kmeans_positive[int(i / split)][i % split] = True
            if not self.kmeans_positive[int(i / split)][i % split]:
                # todo de-activate kmeans in the zone for a while after several failures to speed up
                pass
            i += 1

        self.commit_contours()
        self.metadata["K-means: {} / 9"] = 9 - skipped_kmeans
        self.display_intersections(grid, goban_img)

    def subregions(self, split=3):
        """
        Yield the different (rs, re, cs, ce) indices, each tuple representing a subregion of the goban.
        split -- the number of row split (also used to split columns). Eg. split=3 yields 9 regions before stopping.

        rs -- row start
        re -- row end
        cs -- column start
        ce -- column end

        """
        step = int(gsize / split)
        rs = 0
        re = step
        while re <= gsize:
            cs = 0
            ce = step
            while ce <= gsize:
                yield rs, re, cs, ce
                cs = ce
                ce += step
                if 0 < gsize - ce < step:
                    ce = gsize
            rs = re
            re += step
            if 0 < gsize - re < step:
                re = gsize

    def commit_contours(self):
        for i in range(gsize):
            for j in range(gsize):
                if self.is_empty(i, j):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(self.contour_accu[i][j], return_counts=True)
                    if len(vals) < 3 and E in vals:
                        k = 0 if vals[0] is E else 1
                        if counts[k] / self.histo_len < 0.4:  # if less than 40% empty
                            self.suggest(vals[1 - k], i, j)

    def _learn(self):
        pass

    def _window_name(self):
        return SfMeta.label
