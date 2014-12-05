import cv2
from numpy import zeros, ndarray, unique, uint8, vectorize, max as npmax
from numpy.ma import absolute

from camkifu.core.imgutil import draw_str
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_contours import SfContours
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W, E
from golib.model.exceptions import StateError
from golib.model.move import Move
from golib.model.rules import RuleUnsafe


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
        self.histo_len = 7  # todo integrating a background detector should help lowering that (less false positives)

        # contours-related attributes
        self.contour_accu = zeros((gsize, gsize, self.histo_len), dtype=object)
        self.contour_accu[:] = E

        # cluster-related attributes
        self.cluster_accu = zeros((gsize, gsize, self.histo_len), dtype=object)
        self.cluster_accu[:] = E

    def _find(self, goban_img: ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> dynamically assign one finder per region, periodically trying to introduce clustering where it's not assigned

        """
        self.metadata["Frame nr: {}"] = self.total_f_processed + 1
        if 0 < self.init_frames_left():
            self.warmup(goban_img)
        else:
            self.routine(goban_img)
            self.draw_finders(goban_img)
            self._show(goban_img)

    def init_frames_left(self):
        return self.histo_len - self.total_f_processed

    def warmup(self, img):
        # 1. accumulate a few passes of contours analysis results
        cont_stones = self.contour.find_stones(img)
        self.contour_accu[:, :, self.total_f_processed % self.histo_len] = cont_stones
        self.commit(self.contour_accu)

        # 2. When about to exit warmup, decide which regions are dense enough for clustering finder
        if self.init_frames_left() == 1:
            grid = self.find_intersections(img)
            self.display_intersections(grid, img)
            self.assess_clustering(img, grid)  # todo analyse several frames before making the decisions
            self.commit(self.cluster_accu)
            self._posgrid.learn(absolute(grid))  # do that at the end only, not to invalidate already computed values

    def routine(self, img: ndarray) -> None:
        split = 3
        i = 0
        for rs, re, cs, ce in self.subregions(split=split):
            finder = self.finders[int(i / split)][i % split]
            stones = finder.find_stones(img, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            k = self.total_f_processed % self.histo_len
            self.get_accu(finder)[rs:re, cs:ce, k] = stones[rs:re, cs:ce]
            i += 1
        self.commit(self.contour_accu)
        self.commit(self.cluster_accu)

    def assess_clustering(self, goban_img: ndarray, grid) -> None:
        """
        Try and evaluate "clustering finder" in regions where it is not the default finder.

        """
        # todo switch to global clustering when 4 - 5 regions out of the 9 are clustering ? + at least one per row / col
        split = 3
        ref_stones = self.get_stones()
        i = 0
        # Objective : determine for each region if clustering detection can be applied
        # Basically once a region is dense enough, the clustering seems much more trustworthy than contours analysis
        wild_guess = 0
        for rs, re, cs, ce in self.subregions(split=split):
            # 1. Skipp regions already assigned to clustering
            if self.finders[int(i / split)][i % split] is self.cluster:
                continue

            # 2. Run clustering-based stones detection
            result = self.cluster.find_stones(goban_img, r_start=rs, r_end=re, c_start=cs, c_end=ce)

            # 3. Assess clustering results validity
            passed = 0
            veto = False  # set to True if at least one check was refused
            for constraint in (self.check_logic, self.check_against, self.check_lines):  # todo remove logic from there
                check = constraint(result, reference=ref_stones, grid=grid, r_start=rs, r_end=re, c_start=cs, c_end=ce)
                if 0 <= check:
                    passed += check
                    if check: print("{} passed".format(constraint.__name__))
                    else: print("{} undetermined".format(constraint.__name__))
                else:
                    veto = True
                    print("{} refused".format(constraint.__name__))
                    break

            # 3.2 If at least one test could be run (and passed), assign clustering to that region, and store result
            if not veto and 0 < passed:
                self.cluster_accu[rs:re, cs:ce, self.total_f_processed % self.histo_len] = result[rs:re, cs:ce]
                self.finders[int(i / split)][i % split] = self.cluster
            else:
                # last resort check. separated form others since it's not a real test as it can be undetermined at best
                check = self.check_logic(result, r_start=rs, r_end=re, c_start=cs, c_end=ce)
                if 0 <= check:
                    self.finders[int(i / split)][i % split] = self.cluster
                    wild_guess += 1
            i += 1
        print("Wild guessed regions (warmup): {}/9".format(wild_guess))

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

    def check_against(self, stones, reference=None, r_start=0, r_end=gsize, c_start=0, c_end=gsize, **kwargs):
        """
        Return -1, 0, 1 if the check is respectively refused, undetermined, or passed.

        """
        refs = 0      # number of stones found in reference
        matches = 0   # number of matches with the references (non-empty only)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if reference[r][c] in (B, W):
                    refs += 1
                    if stones[r][c] is reference[r][c]:
                        matches += 1
        if 3 < refs:  # check against at least 4 stones
            if 0.9 < matches / refs:
                return 1  # passed
            else:
                return -1  # refused
        return 0    # undetermined

    def check_lines(self, stones: ndarray, grid: ndarray=None, r_start=0, r_end=gsize, c_start=0, c_end=gsize, **kwargs):
        """
        Check that the provided "stones" 2D array is coherent with lines detection in the image: no line should
        be found in zones where a stone has been detected.

        A confirmation is counted for the zone if it is empty (E) and at least one line has also been detected
        in that zone.

        stones -- a 2D array that can store the objects, used to record the stones found. It is created if not provided.
        grid -- a 3D (or 2D) array that has negative values where lines have been found.

        Return
        lines -- the number of intersections where lines have been detected (based on the provided grid)
        confirms -- the number of empty intersections where lines have been detected, meaning the

        """
        lines = 0    # the number of intersections where lines have been detected
        matches = 0  # the number of empty intersections where lines have been detected
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                if sum(grid[i][j]) < 0:
                    lines += 1
                    if stones[i][j] is E:
                        matches += 1
        if 4 < lines:
            if 0.9 < matches / lines:
                return 1  # passed
            else:
                return -1  # refused
        return 0  # undetermined

    def check_logic(self, stones, r_start=0, r_end=gsize, c_start=0, c_end=gsize, **kwargs):
        """
        Check that the provided "stones" 2D array is coherent with Go logic. This is of course inextricable,
        but some major directions can be checked.

        """
        rule = RuleUnsafe()
        # 1. primal test, see if rules are complaining (suicide for example).
        try:
            move_nr = 1
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if stones[r][c] in (B, W):
                        rule.put(Move('np', (stones[r][c], r, c), number=move_nr), reset=False)
                        move_nr += 1
            rule.confirm()
            # todo check that no kill has happened if we are in warmup phase
        except StateError as se:
            print(se)
            return -1  # refused
        # 2. thick ugly chunks vetoer
        for color in (B, W):
            avatar = vectorize(lambda x: 1 if x is color else 0)(stones.reshape(stones.shape[0] * stones.shape[1]))
            # diagonal moves cost as much as side moves
            dist = cv2.distanceTransform(avatar.reshape(stones.shape).astype(uint8), cv2.DIST_C, 3)
            if 3 < npmax(dist):
                # a stone surrounded by a 3-stones-thick wall of its own color is most likely not Go
                print("{} thick chunk refused".format(color))
                return -1
        # 3. if survived up to here, can't really confirm, but at least nothing seems wrong
        return 0

    def get_accu(self, finder):
        if isinstance(finder, SfClustering):
            return self.cluster_accu
        elif isinstance(finder, SfContours):
            return self.contour_accu

    def draw_finders(self, img):
        split = 3
        i = 0
        for rs, re, cs, ce in self.subregions(split=split):
            x0, y0, _, _ = self._getrect(rs, cs)
            _, _, x1, y1 = self._getrect(re - 1, ce - 1)
            finder = self.finders[int(i / split)][i % split]
            if isinstance(finder, SfClustering):
                draw_str(img[x0:x1, y0:y1], "K")
            elif isinstance(finder, SfContours):
                draw_str(img[x0:x1, y0:y1], "C")
            i += 1

    def _learn(self):
        pass

    def _window_name(self):
        return SfMeta.label
