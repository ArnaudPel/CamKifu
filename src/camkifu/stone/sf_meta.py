import math

import numpy as np
from golib.config.golib_conf import gsize, E

import camkifu.core
from camkifu.core import imgutil

import camkifu.stone
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_contours import SfContours


__author__ = 'Arnaud Peloquin'

# possible states of a region. watch indicate no active search need to be performed
Warmup = "warmup"
Search = "search"
Idle = "idle"


class SfMeta(camkifu.stone.StonesFinder):
    """
    Work in progress.

    A "meta" stones finder at the moment, since it aims to aggregate contributions from the different stones finder
    experimented so far, with the hope to become something actually useful.

    """

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.cluster = SfClustering(None)    # set to None for safety, put vmanager when needed
        self.contour = SfContours(vmanager)  # this one needs a vmanager already
        self.contour._show = self._show  # hack
        self.contour.get_foreground = self.get_foreground  # hack
        self.routine_constr = {
            self.cluster: (self.check_against, self.check_flow),
            self.contour: (self.check_flow,)
        }

        # the whole processing is divided into subregions
        self.split = 3
        self.histo = 3
        # store which finder should be used on each goban region
        self.regions = np.zeros((self.split, self.split), dtype=object)
        for r in range(self.split):
            for c in range(self.split):
                self.regions[r, c] = Region(self, self.subregion(r, c), self.histo, finder=self.contour, state=Warmup)

    def _find(self, goban_img: np.ndarray):
        """
        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> dynamically assign one finder per region, periodically trying to introduce clustering where it's not assigned

        """
        # 0. if startup phase: initialize background model
        if self.total_f_processed < self.bg_init_frames:
            self.display_bg_sampling(goban_img)
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
        except camkifu.core.CorrectionWarning as cw:
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
        return SfMeta.__name__


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
        self.states = imgutil.CyclicBuffer(1, self.histo, dtype=object, init=state)
        self.data = []

        shape = (self.re - self.rs, self.ce - self.cs)
        self.contour_accu = imgutil.CyclicBuffer(shape, self.histo, dtype=object, init=E)
        self.cluster_accu = imgutil.CyclicBuffer(shape, self.histo, dtype=object, init=E)
        self.accus = {self.sf.contour: self.contour_accu, self.sf.cluster: self.cluster_accu}

        self.canvas = None
        self.population = -1  # populations of regions as they  where the last time clustering was assessed
        self.cluster_score = imgutil.CyclicBuffer(1, self.histo, dtype=np.int8)  # assessment score

    def process(self, img: np.ndarray, ref_stones: np.ndarray, canvas: np.ndarray=None) -> None:
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
            kwargs = {'rs': self.rs, 're': self.re, 'cs': self.cs, 'ce': self.ce, 'canvas': canvas}
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
                    tried = False
                    if self.clustering_needs_try(ref_stones):
                        self.try_clustering(img, ref_stones)
                        tried = True
                    # default routine sequence
                    if (not tried) or (self.finder is not self.sf.cluster):
                        self.routine(img, ref_stones, **kwargs)
                    self.states[0] = Idle
            self.states.increment()
        else:
            self.data.append("A")  # agitated
        self.draw_data()

    def routine(self, img, refs, **kwargs):
        """
        Run the current finder on 'img' to detect stones, then verify and commit them.

        """
        stones = self.finder.find_stones(img, **kwargs)
        if stones is not None:
            self.discard_lonelies(stones, refs)
            constraints = self.sf.routine_constr[self.finder]
            passed = self.verify(constraints, stones, refs, img, id_="routine")
            if 0 <= passed:
                accu = self.accus[self.finder]
                accu[:] = stones[self.rs:self.re, self.cs:self.ce]
                self.commit(accu)

    def try_clustering(self, img, refs):
        """
        Run and evaluate "clustering finder" in this region. Basically once a region is dense enough,
        the clustering seems much more trustworthy than contours analysis, so it should be used as soon as possible.

        """
        # Run clustering-based stones detection
        stones = self.sf.cluster.find_stones(img, rs=self.rs, re=self.re, cs=self.cs, ce=self.ce)
        passed = -1
        if stones is not None:
            constraints = (self.sf.check_thickness, self.sf.check_against, self.sf.check_lines)
            passed = self.verify(constraints, stones, refs, img, id_="tryclust")
        # accumulate results
        self.cluster_score[0] = passed
        if 0 < passed:  # store successfully tested scores
            self.cluster_accu[:] = stones[self.rs:self.re, self.cs:self.ce]
            self.commit(self.cluster_accu)
        if self.cluster_score.at_end():  # wait the end of the cycle before making the decision
            total_score = np.sum(self.cluster_score.buffer)
            if 0 <= total_score:
                # accept if at least one more success than failures
                self.finder = self.sf.cluster
                self.states.buffer[:] = Search  # new finder, run a full cycle
                if total_score == 0 and np.max(self.cluster_score.buffer) == 0:
                    print("Wild assignment of clustering to region {}".format((self.re, self.ce)))
                    # else:
                    # print("Clustering vetoed {} in region {}".format(passed, (self.re, self.ce)))
        self.cluster_score.increment()
        self.population = self.get_population(refs)

    def verify(self, constraints, stones, refs, img, id_=""):
        """
        Check that the provided 'constraints' are verified by 'stones'.
        refs -- an array of reference to compare the stones against (eg., the stones found so far)
        img -- the image from which the 'stones' result has been extracted
        id_ -- an optional identifier to add to the beginning of printed messages

        @param constraints:
        """
        passed = 0
        for constr in constraints:
            check = constr(stones, img=img, reference=refs, rs=self.rs, re=self.re, cs=self.cs, ce=self.ce)
            if check < 0:
                if len(id_): id_ += ": "
                print("{}{} vetoed region {}".format(id_, constr.__name__, (self.re, self.ce)))
                passed = -1  # veto from that constraint
                break
            passed += check
        return passed

    def commit(self, cb: imgutil.CyclicBuffer) -> None:
        assert len(cb.buffer.shape) == 3
        moves = []
        for i in range(self.re - self.rs):
            for j in range(self.ce - self.cs):
                if self.sf.is_empty(i + self.rs, j + self.cs):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = np.unique(cb.buffer[i, j], return_counts=True)
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
        except camkifu.core.DeletedError as de:
            print(str(de))
            pass  # would be nice to learn from it someday..
        cb.increment()

    def check_foreground(self) -> bool:
        """
        Watch for foreground disturbances to "wake up" the regions of interest only.
        Return True if the region is "calm" (not much foreground), else False.

        fg -- the global foreground image. expected to contain values equal to 0 or 255 only.

        """
        try:
            fg = self.sf.get_foreground()
        except ValueError:
            return True  # background analysis seems to be disabled, allow all frames.

        # first, check that the outer border of this region is not moving
        moving = 0
        border_threshold = 2  # the min number of agitated border intersections to trigger "agitated"
        for a0, b0, a1, b1 in self.outer_border():
            if (a1-a0) * (b1-b0) * 0.7 < np.sum(fg[a0:a1, b0:b1]) / 255:
                if (a0 == 0 or a1 == fg.shape[0]-1) and (b0 == 0 or b1 == fg.shape[1]-1):
                    moving = border_threshold  # moved at corner, set agitated since it is a limit condition
                moving += 1
                if border_threshold <= moving:
                    self.set_agitated()
                    return False
            # for i in range(3):
            #     self.canvas[a0:a1, b0:b1, i] = fg[a0:a1, b0:b1]

        # then, allow for a few (2-3) new stones to have been detected as foreground (it takes some time to learn)
        x0, y0, x1, y1 = self.get_img_bounds()
        subfg = fg[x0:x1, y0:y1]
        threshold = 3 * (self.sf.stone_radius()**2) * math.pi  # 3 times the expected area of a stone
        agitated = np.sum(subfg) / 255
        if threshold < agitated:
            self.set_agitated()
            return False
        return True

    def set_agitated(self):
        """
        Trigger search state for this region. If still in Warmup state, extend its duration.

        """
        if self.states[0] is Warmup:
            # increase the duration of warmup a bit
            self.states.replace(Idle, Warmup)
        else:
            # trigger full search
            self.states.buffer[:] = Search

    def clustering_needs_try(self, ref_stones) -> bool:
        """
        Indicate whether clustering finder should be tried in that region.
        ref_stones -- the stones found so far (the whole goban), used to determine a reference population.

        """
        current_pop = self.get_population(ref_stones[self.rs:self.re, self.cs:self.ce])
        if current_pop < 4:
            return False
        return (not self.cluster_score.at_start()) or \
            self.finder is not self.sf.cluster and self.population + 1 < current_pop  # try again every 2 new stones

    def get_population(self, stones):
        # noinspection PyTupleAssignmentBalance
        vals, counts = np.unique(stones, return_counts=True)
        return sum([counts[idx] for idx, v in enumerate(vals) if v is not E])

    def get_img_bounds(self):
        """
        Return the rectangle defining the sub-image associated with this region, in relation to the global image.
        Return x0, y0, x1, y1
        x0, y0 -- first point of rectangle
        x1, y1 -- second point of rectangle

        """
        x0, y0, _, _ = self.sf.getrect(self.rs, self.cs)
        _, _, x1, y1 = self.sf.getrect(self.re - 1, self.ce - 1)
        return x0, y0, x1, y1

    def discard_lonelies(self, stones: np.ndarray, reference: np.ndarray):
        """
        Discard all the stones on the first line that have no neighbour in a 2-lines thick square around them.

        """
        lonelies = self.sf.first_line_lonelies(stones, reference, rs=self.rs, re=self.re, cs=self.cs, ce=self.ce)
        for r, c in lonelies:
            # correct that kind of error instead of refusing result, since it may be recurring
            stones[r, c] = E
        if len(lonelies):
            print("Discarded lonely stone(s) on first line {}".format(lonelies))

    def outer_border(self):
        """
        Enumerate the intersections pixels around this region.
        Concretely, yield the pixel rectangles around the sub-image of this region (outside of it), as per
        StonesFinder.getrect(r, c).

        """
        x = max(0, self.cs-1)
        for y in range(max(0, self.rs-1), min(gsize, self.re+1)):
            yield self.sf.getrect(y, x)

        y = min(gsize-1, self.re)
        for x in range(max(1, self.cs), min(gsize, self.ce+1)):
            yield self.sf.getrect(y, x)

        x = min(gsize-1, self.ce)
        for y in range(min(gsize-2, self.re-1), max(-1, self.rs-2), -1):
            yield self.sf.getrect(y, x)

        y = max(0, self.rs-1)
        for x in range(min(gsize-2, self.ce-1), max(0, self.cs-1), -1):
            yield self.sf.getrect(y, x)

    def draw_data(self):
        x0, y0, x1, y1 = self.get_img_bounds()
        if isinstance(self.finder, SfClustering):
            self.data.append("K")
        elif isinstance(self.finder, SfContours):
            self.data.append("C")
        imgutil.draw_str(self.canvas[x0:x1, y0:y1], str(self.data))
        self.data.clear()
