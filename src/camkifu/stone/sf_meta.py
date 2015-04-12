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
    """ Stones finder implementation aggregating contributions from SfClustering and SfContours.

    Sums up what I have been able to put together so far regarding stones detection. There's still (much) more to do
    to have something sensitive and robust, probably a different approach from scratch.

    The frame area is divided into Regions (see class below), each responsible for handling the detection in its area.
    This allows for different finders to be used in different part of the image based on their respective density.

    In order to filter out some false positives, a buffer of recent detection results is maintained. A stone is
    successfully detected if it is "sufficiently present" in that short-lived history.

    Note: because of background sampling and consistency checking (among others), this class is stateful. This
    means things may easily break if the movie is rolled back (using the slider), or another movie is played
    without reseting this finder.

    Args:
        cluster: StonesFinder
            Delegate preferred for mid to high density (stonewise) areas.
        contour: StonesFinder
            Delegate preferred for low density (stonewise) areas.
        routine_constr: dict
            Associate to each stones finder delegate the constraints that should be used to test its results.
        split: int
            Frames are divided into (split * split) Regions.
        histo: int
            The length of the (short-lived) history used to filter out false positives (see above).
        regions: ndarray
            The Region objects, each being responsible for stones detection in its image part.
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
        """ The stones detection main algorithm. Delegate work to each Region and show some results on an image.

        Baselines :
        - trust contours to find single stones. but it will fail to detect all stones in a chain or cluster.
        - trust clustering to find all stones if their density is high enough. but it will fail if too few stones
        -> dynamically assign one finder per region, periodically trying to introduce clustering where it's not assigned

        """
        # 0. if startup phase: let background model get initialized (see StonesFinder._doframe())
        if self.total_f_processed < self.bg_init_frames:
            self.display_bg_sampling(goban_img.shape)
            return
        # 1. delegate all the work to Regions
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
        """ Get the row and column indices representing the requested subregion of the goban.

        Ex if split is 3, the region (0, 1) has row indices (0, 6), and column indices (6, 12).

        Args:
            row: int
            col: int
                The identifiers of the region, in [ 0, self.split [

        Returns rs, re, cs, ce:  int, int, int, int
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
    """ Manage stones detection processes in a defined subregion of the Goban image.

    This approach enables different regions to be processed by different stones finder depending on their stones
    density.

    On background segmentation:
    Regions that have too much foreground are considered "agitated", they don't run search until things calm down.
    Inversely, regions that have been quiet and searched during enough frames are put to idle state until motion
    is detected.

    Attributes:
        sf: StonesFinder
            Provides access to stones finder util methods.
        rs, re, cs, ce: int, int, int, int
            The row and column boundaries (in terms of Goban intersections) where this Region should search stones.
        histo: int
            The length of the (short-lived) history used to filter out false positives.
        finder: StonesFinder
            The delegate currently responsible for stones detection in this region.
        states: CyclicBuffer
            A cyclic history of the latest states this Region was in.
        data: list
            Strings to display on an image for that region (state, foreground status, search method)
        contours_accu: CyclicBuffer
            Accumulator of results from the "contours" stones finder.
        cluster_accu: CyclicBuffer
            Accumulator of results from the "clustering" stones finder.
        accus: dict
            Mapping of StonesFinders to their respective accumulator.
        canvas: ndarray
            An optional image where to draw some algorithms results.
        population: int
            The number of stones found in this region so far.
        cluster_score: CyclicBuffer
            A cyclic history of the latest clustering assessment scores.
    """

    def __init__(self, sf: SfMeta, boundaries: tuple, histo: int, finder=None, state: str=None):
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
        self.population = -1
        self.cluster_score = imgutil.CyclicBuffer(1, self.histo, dtype=np.int8)  # assessment score

    def process(self, img: np.ndarray, ref_stones: np.ndarray, canvas: np.ndarray=None) -> None:
        """ Run stones detection in the region of img corresponding to this Region.

        The main drive of this implementation is to be able to switch from SfContours to SfClustering when the stones
        population becomes high enough is this region. The fact that the Goban may already be populated at startup
        should also be acknowledged (hence the introduction of the 'warmup' phase).

        Args:
            img: ndarray
                The global goban img. The part corresponding to this region will be sliced automatically.
            foreground: ndarray
                The global foreground mask.
        ref_stones
            The global stones reference (the current known state of the whole goban).
        canvas: ndarray
            A global image where to draw things to show to the user.

        Return: None
            Results are directly submitted to the controller as they are obtained.
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
            # 2. if routine phase, record changes (which are not expected to exceed 2-3 moves)
            else:
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
            self.data.append("Q")  # quiet (calm)
            self.states.increment()
        else:
            self.data.append("A")  # agitated
        self.draw_data()

    def routine(self, img, refs, **kwargs) -> None:
        """ Run stones detection on 'img', verify the result, and commit it.
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

    def try_clustering(self, img, refs) -> None:
        """ Set "clustering finder" as the current finder of this region if it returns valid results.

        Motivation: once a region is dense enough, the clustering algorithm seems much more trustworthy
        than contours analysis. So it should be tried often, in order to know as soon as possible when it
        becomes pertinent to use it.
        """
        # Run clustering-based stones detection
        stones = self.sf.cluster.find_stones(img, rs=self.rs, re=self.re, cs=self.cs, ce=self.ce)
        passed = -1
        if stones is not None:
            constraints = (self.sf.check_thickness, self.sf.check_against, self.sf.check_lines)
            passed = self.verify(constraints, stones, refs, img, id_="tryclust")
        # accumulate results
        self.cluster_score[0] = passed
        if 0 < passed:  # commit verified results (despite the fact that is not the current finder)
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
                    #     print("Clustering vetoed {} in region {}".format(passed, (self.re, self.ce)))
        self.cluster_score.increment()
        self.population = self.get_population(refs)

    def verify(self, constraints, stones, refs, img, id_="") -> int:
        """ Check that the provided 'constraints' are verified by 'stones'.

        Args:
            constraints: list
                Test successively, break at the first constraint that vetoes.
            stones: ndarray
                The result to verify.
            refs: ndarray
                The reference (eg. stones found so far) against which to compare 'stones'.
            img: ndarray
                The image from which the 'stones' result has been extracted.
            id_: str
                An optional identifier to add to the beginning of printed messages.

        Returns passed: int
            The sum of constraint scores, or -1 if a constraint has vetoed.
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
        """ Submit results to the controller based on recent history.

        Motivation: false positives are regularly triggered by the stones detection algos. Keeping a
        recent history enables to only submit results that are recurrent, thus less likely to be pollution.

        Args:
            cb: CyclicBuffer
                The "recent history" in which to detect valid new moves.

        Returns: None
            Submit directly to the controller.
        """
        assert len(cb.buffer.shape) == 3
        moves = []
        for i in range(self.re - self.rs):
            for j in range(self.ce - self.cs):
                if self.sf.is_empty(i + self.rs, j + self.cs):
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = np.unique(cb.buffer[i, j], return_counts=True)
                    # consider commit if only one of the (B, W) colors has been triggered
                    if len(vals) == 2 and E in vals:
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
            pass  # the real objective of the exception is to learn from it. someday..
        cb.increment()

    def check_foreground(self) -> bool:
        """ Analyse and react to foreground disturbances.

        The analysis has two distinct phases. First, active foreground at the border (and even more so at the corners)
        may come from a bigger disturbance in a neighbour region. This is why detection must be very sensitive
        in these areas. Second, if the border is 'calm', a more approximate detection can be run on the whole region.

        In case of agitation, the state of this Region is updated in order to trigger future search(es).

        Returns agitated: bool
            True if the region is "calm" (not much active foreground), else False.
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
                    moving = border_threshold  # moved at corner, mark as 'agitated' right away
                moving += 1
                if border_threshold <= moving:
                    self.set_agitated()
                    return False
                    # for i in range(3):
                    #     self.canvas[a0:a1, b0:b1, i] = fg[a0:a1, b0:b1]

        # then, allow for a few (2-3) new stones to be agitating the whole region
        x0, y0, x1, y1 = self.get_img_bounds()
        subfg = fg[x0:x1, y0:y1]
        threshold = 3 * (self.sf.stone_radius()**2) * math.pi  # 3 times the expected area of a stone
        agitated = np.sum(subfg) / 255
        if threshold < agitated:
            self.set_agitated()
            return False
        return True

    def set_agitated(self) -> None:
        """ Trigger search for the next history cycle of this Region.
        """
        if self.states[0] is Warmup:
            # increase the duration of warmup a bit
            self.states.replace(Idle, Warmup)
        else:
            # trigger full search
            self.states.buffer[:] = Search

    def clustering_needs_try(self, ref_stones) -> bool:
        """ Indicate whether clustering finder should be tried in this Region.

        Args:
            ref_stones: ndarray
                The stones found so far (the whole goban), used to determine a reference population.

        Returns: bool
        """
        current_pop = self.get_population(ref_stones[self.rs:self.re, self.cs:self.ce])
        if current_pop < 4:
            return False
        return (not self.cluster_score.at_start()) or \
               self.finder is not self.sf.cluster and self.population + 1 < current_pop  # try again every 2 new stones

    def get_population(self, stones) -> int:
        """ Return the number of stones that have been found so far in this Region.
        """
        # noinspection PyTupleAssignmentBalance
        vals, counts = np.unique(stones, return_counts=True)
        return sum([counts[idx] for idx, v in enumerate(vals) if v is not E])

    def get_img_bounds(self) -> (int, int, int, int):
        """ Get the rectangle defining the sub-image (of the Goban image) represented by this region.

        Returns x0, y0, x1, y1:   int, int, int, int
            x0, y0:  first point of the rectangle
            x1, y1:  second point of the rectangle
        """
        x0, y0, _, _ = self.sf.getrect(self.rs, self.cs)
        _, _, x1, y1 = self.sf.getrect(self.re - 1, self.ce - 1)
        return x0, y0, x1, y1

    def discard_lonelies(self, stones: np.ndarray, reference: np.ndarray) -> None:
        """ Discard all the stones on the first line that are likely to be pollution.

        Stones on the first line that have no neighbour in a 2-lines thick square around them do not look like a real
        Go move. Discard them.
        """
        lonelies = self.sf.first_line_lonelies(stones, reference, rs=self.rs, re=self.re, cs=self.cs, ce=self.ce)
        for r, c in lonelies:
            # correct that kind of error instead of refusing result, since it may be recurring
            stones[r, c] = E
        if len(lonelies):
            print("Discarded lonely stone(s) on first line {}".format(lonelies))

    def outer_border(self):
        """ Yield the sub-image of each intersection composing the outer border of this region.

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

    def draw_data(self) -> None:
        """ Draw and clear the data that has been accumulated by this region so far.
        """
        x0, y0, x1, y1 = self.get_img_bounds()
        if isinstance(self.finder, SfClustering):
            self.data.append("K")
        elif isinstance(self.finder, SfContours):
            self.data.append("C")
        imgutil.draw_str(self.canvas[x0:x1, y0:y1], str(self.data))
        self.data.clear()
