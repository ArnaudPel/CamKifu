from numpy import zeros, ndarray, unique, int8, uint8, max as npmax, sum as npsum

from camkifu.core.exceptions import DeletedError, CorrectionWarning
from camkifu.core.imgutil import draw_str, CyclicBuffer
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_contours import SfContours
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B, W, E


__author__ = 'Arnaud Peloquin'

# possible states of a region. watch indicate no active search need to be performed
Warmup = "warmup"
Search = "search"
Idle = "idle"


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
        self.contour._show = self._show  # hack

        # the whole processing is divided into subregions
        self.split = 3
        self.histo = 3
        self.regions = zeros((self.split, self.split), dtype=object)  # which finder should be used on each goban region
        for r in range(self.split):
            for c in range(self.split):
                self.regions[r, c] = Region(self, self.subregion(r, c), self.histo, finder=self.contour, state=Warmup)

    def _find(self, goban_img: ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> dynamically assign one finder per region, periodically trying to introduce clustering where it's not assigned

        """
        self.intersections = None  # reset cache
        self.metadata["Frame nr: {}"] = self.total_f_processed + 1
        # 0. if startup phase: initialize background model
        if self.total_f_processed < self.bg_init_frames:
            black = zeros((goban_img.shape[0], goban_img.shape[1]), dtype=uint8)
            message = "BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames)
            draw_str(black, message)
            self._show(black)
            return
        else:
            ref_stones = self.get_stones()
            canvas = goban_img.copy()
            for r in range(self.split):
                for c in range(self.split):
                    self.regions[r, c].process(goban_img, ref_stones, canvas=canvas)
            self._show(canvas)

    def _learn(self):
        try:
            return super()._learn()
        except CorrectionWarning as cw:
            # would be nice to use that information someday
            print(str(cw))

    def subregion(self, row, col):
        """
        Return the (rs, re, cs, ce) indices, representing the requested (row, col) subregion of the goban.
        split -- the number of row split (also used to split columns). Eg. split=3 yields 9 regions before stopping.

        rs -- row start     (inclusive)
        re -- row end       (exclusive)
        cs -- column start  (inclusive)
        ce -- column end    (exclusive)

        """
        assert 0 <= row < self.split and 0 <= col < self.split
        step = int(gsize / self.split)
        re = (row + 1) * step
        if gsize - re < step:
            re = gsize
        ce = (col + 1) * step
        if gsize - ce < step:
            ce = gsize
        return row * step, re, col * step, ce

    def _window_name(self):
        return SfMeta.label


class Region():

    def __init__(self, sf: SfMeta, boundaries, histo, finder=None, state=None):
        """
        sf -- the stones finder managing this region.
        histo -- the max number of frames over which data should be accumulated (history, memory)

        """
        self.sf = sf
        self.rs, self.re, self.cs, self.ce = boundaries
        self.histo = histo
        self.finder = finder
        self.states = CyclicBuffer(1, self.histo, dtype=object, init=state)
        self.data = []

        shape = (self.re - self.rs, self.ce - self.cs)
        self.contour_accu = CyclicBuffer(shape, self.histo, dtype=object, init=E)
        self.cluster_accu = CyclicBuffer(shape, self.histo, dtype=object, init=E)
        self.accus = {self.sf.contour: self.contour_accu, self.sf.cluster: self.cluster_accu}

        self.canvas = None
        self.population = -1  # populations of regions as they  where the last time clustering was assessed
        self.cluster_score = CyclicBuffer(1, self.histo, dtype=int8)  # assessment score

    def process(self, img: ndarray, ref_stones: ndarray, canvas: ndarray=None) -> None:
        """
        img -- the global goban img. it will be sliced automatically down below
        foreground -- the global foreground mask. it will be sliced automatically down below
        ref_stones -- the global stones reference (the current known state of the whole goban)
        canvas -- a global image where to draw things to show to the user

        """
        self.canvas = canvas
        self.data.append(self.states[0][0])  # append first letter of current state
        calm = self.check_foreground()
        if calm:
            kwargs = {'r_start': self.rs, 'r_end': self.re, 'c_start': self.cs, 'c_end': self.ce, 'canvas': canvas}
            # 1. if warmup phase: accumulate a few passes of contours analysis
            if self.states[0] is Warmup:
                self.contour_accu[:] = self.sf.contour.find_stones(img, **kwargs)[self.rs:self.re, self.cs:self.ce]
                self.commit(self.contour_accu)
                self.states[0] = Search
            # 2. if routine phase, record changes (which should not exceed 2-3 moves at a time really)
            else:
                self.data.append("Q")  # quiet (calm)
                if self.states[0] is Search:
                    # try clustering if needs be
                    subrefs = ref_stones[self.rs:self.re, self.cs:self.ce]
                    tried = False
                    if self.clustering_needs_try(subrefs):
                        self.try_clustering(img, subrefs)
                        tried = True
                    # default routine sequence
                    if (not tried) or (self.finder is not self.sf.cluster):
                        stones = self.finder.find_stones(img, **kwargs)
                        if stones is not None:
                            # todo check that no more than 2-3 stones have been added
                            # todo if multiple new stones, check both colors are equally present
                            accu = self.accus[self.finder]
                            accu[:] = stones[self.rs:self.re, self.cs:self.ce]
                            self.commit(accu)
                    self.states[0] = Idle
            self.states.increment()
        else:
            self.data.append("A")  # agitated
        self.draw_data()

    def try_clustering(self, img, subrefs):
        """
        Run and evaluate "clustering finder" in this region. Basically once a region is dense enough,
        the clustering seems much more trustworthy than contours analysis, so it should be used as soon as possible.

        """
        # Run clustering-based stones detection
        stones = self.sf.cluster.find_stones(img, r_start=self.rs, r_end=self.re, c_start=self.cs, c_end=self.ce)
        passed = -1
        if stones is not None:
            substones = stones[self.rs:self.re, self.cs:self.ce]
            # Check some validity constraints
            for constraint in (self.sf.check_logic, self.sf.check_against, self.sf.check_lines):
                check = constraint(substones, img=img, reference=subrefs, rs=self.rs, cs=self.cs)
                if check < 0:
                    print("{} vetoed region {}".format(constraint.__name__, (self.re, self.ce)))
                    passed = -1  # veto from that constraint
                    substones = None
                    break
                passed += check
        # accumulate results
        self.cluster_score[0] = passed
        if 0 < passed:  # store successfully tested scores
            self.cluster_accu[:] = substones
            self.commit(self.cluster_accu)
        if self.cluster_score.at_end():  # wait the end of the cycle before making the decision
            total_score = npsum(self.cluster_score.buffer)
            if 0 <= total_score:
                # accept if at least one more success than failures
                self.finder = self.sf.cluster
                self.states.buffer[:] = Search  # new finder, run a full cycle
                if total_score == 0 and npmax(self.cluster_score.buffer) == 0:
                    print("Wild assignment of clustering to region {}".format((self.re, self.ce)))
                    # else:
                    # print("Clustering vetoed {} in region {}".format(passed, (self.re, self.ce)))
        self.cluster_score.increment()
        self.population = self.get_population(subrefs)

    def commit(self, cb: CyclicBuffer) -> None:
        assert len(cb.buffer.shape) == 3
        moves = []
        for i in range(self.re - self.rs):
            for j in range(self.ce - self.cs):
                if self.sf.is_empty(i + self.rs, j + self.cs):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(cb.buffer[i, j], return_counts=True)
                    if len(vals) == 2 and E in vals:  # don't commit if the two colors have been triggered
                        k = 0 if vals[0] is E else 1
                        if counts[k] / cb.size < 0.4:  # commit if less than 40% empty
                            moves.append((vals[1 - k], i + self.rs, j + self.cs))
                    elif len(vals) == 1 and E not in vals:
                        moves.append((vals[0], i + self.rs, j + self.cs))
        try:
            if 1 < len(moves):
                self.sf.bulk_update(moves)
            elif len(moves):
                self.sf.suggest(*moves[0], doprint=False)
        except DeletedError as de:
            print(str(de))
            pass  # would be nice to learn from it someday..
        cb.increment()

    def check_foreground(self):
        """
        Watch for foreground disturbances to "wake up" the regions of interest only.
        Return True if the region is "calm" (not much foreground), else False.

        fg -- the global foreground image. expected to contain values equal to 0 or 255 only.

        """
        try:
            fg = self.sf.get_foreground()
        except ValueError:
            return True  # background analysis seems to be disabled, allow all frames.
        x0, y0, x1, y1 = self.get_subimg_bounds()
        subfg = fg[x0:x1, y0:y1]
        threshold = subfg.shape[0] * subfg.shape[1] / (self.re - self.rs) / (self.ce - self.cs)
        agitated = npsum(subfg) / 255
        if threshold * 0.9 < agitated:
            # equivalent of one intersection moved 90%. trigger search state and mark as agitated
            if self.states[0] is Warmup:
                # increase the duration of warmup a bit
                self.states.replace(Idle, Warmup)
            else:
                # trigger full search
                self.states.buffer[:] = Search
            return False
        return True

    def clustering_needs_try(self, ref_stones) -> bool:
        """
        Indicate whether clustering finder should be tried in that region.
        stones -- the stones of the region (and NOT the whole goban), used to determine a reference population.

        """
        current_pop = self.get_population(ref_stones)
        return (not self.cluster_score.at_start()) or \
            self.finder is not self.sf.cluster and self.population + 1 < current_pop  # try again every 2 new stones

    def get_population(self, stones):
        # noinspection PyTupleAssignmentBalance
        vals, counts = unique(stones, return_counts=True)
        return sum([counts[idx] for idx, v in enumerate(vals) if v is not E])

    def get_subimg_bounds(self):
        x0, y0, _, _ = self.sf.getrect(self.rs, self.cs)
        _, _, x1, y1 = self.sf.getrect(self.re - 1, self.ce - 1)
        return x0, y0, x1, y1

    def draw_data(self):
        x0, y0, x1, y1 = self.get_subimg_bounds()
        if isinstance(self.finder, SfClustering):
            self.data.append("K")
        elif isinstance(self.finder, SfContours):
            self.data.append("C")
        draw_str(self.canvas[x0:x1, y0:y1], str(self.data))
        self.data.clear()
