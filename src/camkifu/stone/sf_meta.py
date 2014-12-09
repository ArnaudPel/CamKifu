from collections import defaultdict

import cv2
from numpy import zeros, ndarray, unique, uint8, vectorize, max as npmax, sum as npsum

from camkifu.core.imgutil import draw_str, CyclicBuffer
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_contours import SfContours
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W, E
from golib.model.exceptions import StateError
from golib.model.move import Move
from golib.model.rules import RuleUnsafe


__author__ = 'Arnaud Peloquin'

# possible states of a region. watch indicate no active search need to be performed
Search = "search"
Watch = "watch"


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
        self.split = 3

        self.finders = zeros((self.split, self.split), dtype=object)  # which finder should be used on each goban region
        self.finders[:] = self.contour  # contour analysis is more suitable for initial phase

        self.histo_len = 3  # the max number of frames over which data should be accumulated (history, memory)
        self.states = CyclicBuffer((self.split, self.split), self.histo_len, dtype=object, init=Search)

        # contours-related attributes
        self.contour_accu = CyclicBuffer((gsize, gsize), self.histo_len, dtype=object, init=E)
        self.contour._show = self._show  # hack

        # cluster-related attributes
        self.cluster_accu = CyclicBuffer((gsize, gsize), self.histo_len, dtype=object, init=E)
        self.cluster_totry = zeros((self.split, self.split), dtype=bool)  # regions where clustering must be assessed
        self.cluster_totry[:] = True
        self.cluster_score = CyclicBuffer((self.split, self.split), self.histo_len, dtype=uint8)  # assessment score

        # background-related attributes
        self.bg_model = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.bg_init_frames = 50

        # dev attributes
        self.region_data = defaultdict(list)  # one list of data per region. acces with str key: "row-index"

    def _find(self, goban_img: ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> dynamically assign one finder per region, periodically trying to introduce clustering where it's not assigned

        """
        self.metadata["Frame nr: {}"] = self.total_f_processed + 1
        learning = 0.01 if self.total_f_processed < self.bg_init_frames else 0.005
        fg = self.bg_model.apply(goban_img, learningRate=learning)  # learn background and get foreground
        # 0. if startup phase: initialize background model
        if self.total_f_processed < self.bg_init_frames:
            black = zeros((goban_img.shape[0], goban_img.shape[1]), dtype=uint8)
            message = "BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames)
            draw_str(black, message, int(black.shape[0] / 2 - 70), int(black.shape[1] / 2))
            self._show(black)
            return
        # 1. if warmup phase: accumulate a few passes of contours analysis
        if 0 < self.warmup_frames_left():
            cont_stones = self.contour.find_stones(goban_img, show=True)
            self.contour_accu[:] = cont_stones
            self.commit(self.contour_accu)
        # 2. if routine phase, record changes (which should not exceed 2-3 moves at a time, if players are really fast)
        else:
            self.routine(goban_img, fg)
            self.draw_regdata(goban_img)
            self._show(goban_img)
            self.states.increment()

    def warmup_frames_left(self):
        return self.bg_init_frames + self.histo_len - self.total_f_processed

    def routine(self, img: ndarray, foreground: ndarray) -> None:
        i = 0
        grid = None  # the intersections detected in current image.
        to_commit = set()
        updated_score = False
        for rs, re, cs, ce in self.subregions(split=self.split):
            row = int(i / self.split)
            col = i % self.split
            calm = self.check_foreground(foreground, i, rs, re, cs, ce)  # todo handle that "i" better, or document it
            state = self.states[row, col]
            self.region_data["{}-{}".format(row, col)].append(state[0])
            if calm:
                self.region_data["{}-{}".format(row, col)].append("Q")  # quiet (calm)
                if state is Search:
                    if self.cluster_totry[row, col] and self.finders[row, col] is not self.cluster:
                        if grid is None:
                            grid = self.find_intersections(img)
                        passed, stones = self.assess_clustering(img, grid, i, rs, re, cs, ce)
                        if 0 <= passed:
                            self.cluster_score[row, col] = passed
                            updated_score = True
                            if self.cluster_score.index % self.histo_len == self.histo_len - 1:  # todo implement correctly
                                if 0 < npsum(self.cluster_score.buffer[row, col, :]):
                                    # at least one test could be run and passed
                                    self.finders[row, col] = self.cluster
                                    self.cluster_accu[rs:re, cs:ce] = stones[rs:re, cs:ce]
                                    to_commit.add(self.cluster_accu)
                                    self.states.buffer[row, col, :] = Search  # new finder, need to run a full cycle
                                self.cluster_totry[row, col] = False
                        else:  # one veto, cancel this "clustering try" cycle, it is too early (not enough stones)
                            self.cluster_totry[row, col] = False
                    else:
                        # default routine code
                        stones = self.finders[row, col].find_stones(img, r_start=rs, r_end=re, c_start=cs, c_end=ce)
                        cbuff = self.get_accu(self.finders[row, col])
                        cbuff[rs:re, cs:ce] = stones[rs:re, cs:ce]
                        to_commit.add(cbuff)
                    self.states[row, col] = Watch
            else:
                self.region_data["{}-{}".format(row, col)].append("A")  # agitated
            i += 1
        for cb in to_commit:
            self.commit(cb)
        if updated_score:
            self.cluster_score.increment()

    def assess_clustering(self, goban_img: ndarray, grid: ndarray, i, rs=0, re=gsize, cs=0, ce=gsize) -> ndarray:
        """
        Try and evaluate "clustering finder" in regions where it is not the default finder.
        Basically once a region is dense enough, the clustering seems much more trustworthy than contours analysis.

        """
        # Run clustering-based stones detection
        stones = self.cluster.find_stones(goban_img, r_start=rs, r_end=re, c_start=cs, c_end=ce)

        # Assess clustering results validity
        ref_stones = self.get_stones()
        passed = 0
        for constraint in (self.check_against, self.check_lines, self.check_logic):
            check = constraint(stones, reference=ref_stones, grid=grid, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            if check < 0:
                return -1, None  # veto from that constraint check
            passed += check
        row, col = int(i / self.split), i % self.split
        if not passed:
            print("Wild clustering assignment to region {}".format((row, col)))
        return passed, stones

    def check_foreground(self, fg: ndarray, i, rs=0, re=gsize, cs=0, ce=gsize):
        """
        Watch for foreground disturbances to "wake up" the regions of interest only.
        Return True if the region is "calm" (not much foreground), else False.

        """
        x0, y0, _, _ = self._getrect(rs, cs)
        _, _, x1, y1 = self._getrect(re - 1, ce - 1)
        threshold = (x1 - x0) * (y1 - y0) / (re - rs) / (ce - cs)
        agitated = npsum(fg[x0:x1, y0:y1]) / 255  # fg array expected to contain 0 or 255 values only
        if threshold * 0.9 < agitated:
            # equivalent of one intersection moved 90%. trigger search state and mark as agitated
            self.states.buffer[int(i / self.split), i % self.split, :] = Search
            return False
        return True

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

    def commit(self, cb: CyclicBuffer):
        assert len(cb.buffer.shape) == 3
        for i in range(gsize):
            for j in range(gsize):
                if self.is_empty(i, j):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(cb.buffer[i, j], return_counts=True)
                    if len(vals) < 3 and E in vals:  # don't commit if the two colors have been triggered
                        k = 0 if vals[0] is E else 1
                        if counts[k] / cb.size < 0.4:  # commit if less than 40% empty
                            self.suggest(vals[1 - k], i, j)
        cb.increment()

    def check_against(self, stones, reference=None, r_start=0, r_end=gsize, c_start=0, c_end=gsize, **kwargs):
        """
        Return -1, 0, 1 if the check is respectively refused, undetermined, or passed.

        """
        refs = 0      # number of stones found in reference
        matches = 0   # number of matches with the references (non-empty only)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if reference[r, c] in (B, W):
                    refs += 1
                    if stones[r, c] is reference[r, c]:
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
                if sum(grid[i, j]) < 0:
                    lines += 1
                    if stones[i, j] is E:
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
                    if stones[r, c] in (B, W):
                        rule.put(Move('np', (stones[r, c], r, c), number=move_nr), reset=False)
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
        # todo check that there is no lonely stone on first line (no neighbour say 3 lines around it)
        # 3. if survived up to here, can't really confirm, but at least nothing seems wrong
        return 0

    def get_accu(self, finder):
        # todo replace w a dict ?
        if isinstance(finder, SfClustering):
            return self.cluster_accu
        elif isinstance(finder, SfContours):
            return self.contour_accu

    def draw_regdata(self, img):
        i = 0
        for rs, re, cs, ce in self.subregions(split=self.split):
            x0, y0, _, _ = self._getrect(rs, cs)
            _, _, x1, y1 = self._getrect(re - 1, ce - 1)
            row = int(i / self.split)
            col = i % self.split
            finder = self.finders[row, col]
            data_list = self.region_data["{}-{}".format(row, col)]
            if isinstance(finder, SfClustering):
                data_list.append("K")
            elif isinstance(finder, SfContours):
                data_list.append("C")
            draw_str(img[x0:x1, y0:y1], str(data_list))
            data_list.clear()
            i += 1

    def _learn(self):
        pass

    def _window_name(self):
        return SfMeta.label
