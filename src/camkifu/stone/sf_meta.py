from numpy import uint8, float32, zeros, ndarray, unique
import cv2
from camkifu.core.imgutil import draw_str
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
        self.cluster = SfClustering(None)    # set to None for safety, put vmanager when needed
        self.contour = SfContours(vmanager)  # this one needs a vmanager already

        self.finders = zeros((3, 3), dtype=object)  # which finder should be used on each goban region
        self.finders[:] = self.contour  # contour analysis is more suitable for initial phase
        self.histo_len = 10  # todo integrating a background detector should help lowering that (less false positives)

        # contours-related attributes
        self.contour_accu = zeros((gsize, gsize, self.histo_len), dtype=object)
        self.contour_accu[:] = E
        self.nblines_warmup = None

        # cluster-related attributes
        self.cluster_accu = zeros((gsize, gsize, self.histo_len), dtype=object)
        self.cluster_accu[:] = E

    def _find(self, goban_img: ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> assess clustering result with contours results, and complete with the former if validated

        """
        if 0 < self.init_frames_left():
            self.warmup(goban_img)
        else:
            self.routine(goban_img)

    def init_frames_left(self):
        return self.histo_len - self.total_f_processed

    def warmup(self, img):
        # 1. accumulate a few passes of contours analysis results
        cont_stones = self.contour.find_stones(img)
        self.contour_accu[:, :, self.total_f_processed % self.histo_len] = cont_stones
        self.commit(self.contour_accu)

        # 2. get an idea of the state of the game by accumulating the number of lines detected in image
        lines = 0
        grid = self.find_intersections(img)
        for r in range(gsize):
            for c in range(gsize):
                if sum(grid[r][c]) < 0:
                    lines += 1
        if self.nblines_warmup is None:
            self.nblines_warmup = lines
        else:
            self.nblines_warmup += lines
            self.nblines_warmup /= 2
        self.metadata["Frame nr: {}"] = self.total_f_processed
        self.display_intersections(grid, img)

        # 3. When about to exit warmup, decide which regions are dense enough for clustering finder
        if self.init_frames_left() == 1:
            # if self.nblines_warmup < 0.2 * gsize ** 2:
            #     there should be enough stones on the goban overall for a global clustering
                # self.finders[:] = self.cluster
                # todo check against stones found so far for safety
            # else:
            if self.nblines_warmup < 0.7 * gsize ** 2:
                self.assess_clustering(img, grid)
                self.commit(self.cluster_accu)
            # else: too many lines, don't try clustering now.
            print(self.finders)

    def assess_clustering(self, goban_img: ndarray, grid) -> None:
        """
        Try and evaluate clustering finder in regions where it is not the default finder.

        """
        # todo switch to global clustering when 4 - 5 regions out of the 9 are clustering ? + at least one per row / col
        split = 3
        img = goban_img.astype(float32)
        ref_stones = self.get_stones()
        i = 0
        # Objective : determine for each region if clustering detection can be applied
        # Basically once a region is dense enough, the clustering seems much more trustworthy than contours analysis
        ignored = 0
        for rs, re, cs, ce in self.subregions(split=split):
            if self.finders[int(i / split)][i % split] is self.cluster:
                continue

            # 1. See if we have enough reference values to cross-check clustering's work
            references_count = 0    # total number of stones found by contour analysis
            lines = 0       # total number of lines found by hough
            for r in range(rs, re):
                for c in range(cs, ce):
                    stone = ref_stones[r][c]
                    if stone in (B, W):
                        references_count += 1
                    if sum(grid[r][c]) < 0:
                        lines += 1

            # 2. Run clustering-based stones detection
            ratios, centers = self.cluster.cluster_colors(img, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            clust_result = self.cluster.interpret_ratios(ratios, centers)

            # 3. Assess results from clustering
            score = 0
            # 3.1 Check of clustering against the stones reference
            matches_r = 0
            for r in range(rs, re):
                for c in range(cs, ce):
                    if ref_stones[r][c] in (B, W) and clust_result[r][c] is ref_stones[r][c]:
                        matches_r += 1
            if 4 < references_count and 0.9 < matches_r / references_count:
                score += matches_r

            # 3.2 Check clustering against lines analysis
            lines, matches_l = self.cluster.check_lines(clust_result, grid, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            if 3 < lines and 0.9 < matches_l / lines:
                score += matches_l

            # 3.3 If results have successfully been checked, store result and assign clustering to that region
            if 9 < score:
                self.cluster_accu[rs:re, cs:ce, self.total_f_processed % self.histo_len] = clust_result[rs:re, cs:ce]
                self.finders[int(i / split)][i % split] = self.cluster
            else:
                ignored += 1
                # todo as a last resort, check pertinence based on go rules ? (ignore big ugly chunks)
                pass
            i += 1
        print("Ignored regions (warmup): {} / 9".format(ignored))

    def routine(self, img: ndarray) -> None:
        draw_str(img, "TODO IMPLEMENT MAIN ROUTINE")
        self._show(img)
        # split = 3
        # for rs, re, cs, ce in self.subregions(split=split):
        #     pass

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

    def commit(self, stones):
        assert len(stones.shape) == 3
        for i in range(gsize):
            for j in range(gsize):
                if self.is_empty(i, j):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(stones[i][j], return_counts=True)
                    if len(vals) < 3 and E in vals:  # don't commit if the two colors have been triggered
                        k = 0 if vals[0] is E else 1
                        if counts[k] / stones.shape[2] < 0.4:  # commit if less than 40% empty
                            self.suggest(vals[1 - k], i, j)

    def _learn(self):
        pass

    def _window_name(self):
        return SfMeta.label
