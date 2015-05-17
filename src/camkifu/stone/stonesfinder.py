import math
import queue
import time

import cv2
import numpy as np
import golib.model

import camkifu.core
from camkifu.core import imgutil
from camkifu.config import cvconf

from golib.config.golib_conf import gsize, E, B, W


correc_size = 10


class StonesFinder(camkifu.core.VidProcessor):
    """ Abstract base structure for stones-finding processes.

    Relies on the providing of a transform matrix that enables extracting goban pixels from the global frame.

    Please note that interactions with the Goban (eg. checking emptiness of a location) may happen in a concurrent
    way. Issues may arise from that fact. For example, the detection process (here) may ask for emptiness from the
    stones finder thread, while the Goban status is being updated from the GUI event thread (either by processing an
    earlier instruction issued by this stones finder, or a user action).

    Features:
        Submission of results:
            Utility methods are provided that can help with the communication of results to the controller:
                ¤ suggest(...)      - indicate that a new stone has been played.
                ¤ remove(...)       - indicate that a stone has been removed from the Goban.
                ¤ bulk_update(...)  - indicate that multiple stones have been played or removed.
        Background learning:
            A background model is automatically maintained by default. It provides the ability to do a
            background/foreground segmentation. New frames are automatically submitted to it as soon as they are
            computed, and the resulting foreground mask is available through self.get_foreground().
        Lines detection:
            A line detection feature has been added to help find Goban intersections that are most likely empty.
            Since this detection is not perfect, it should be used as an probabilistic indicator: lines may not be
            detected when they should, or sometimes lines over stones (at the tangent) may appear, although less
            frequently.
        Consistency checking:
            In an attempt to validate the stones detection result by several different approaches, "constraints"
            checking methods have been introduced. They may veto, accept or ignore a stones detection result based
            on the extra data they use (lines detection, history, thickness). These implementations are here as an
            example, although I am not satisfied with their actual value (As of 13/02/15).
        Grid position learning:
            StoneFinder relies on a PosGrid object to manage the location of each Goban intersection. When having
            run lines detection on the current image, the results can be further used by computing the intersection
            of each "cross" found in a given intersection area. By cross understand at least one horizontal and
            one vertical line. This is handled by self.update_grid().
        User corrections learning:
            Detection may fail in several ways, like missing a stone or finding non-existing stones. While the first
            case can be corrected manually without much trouble, the second is more tricky. Indeed, the stone that
            has been wrongly detected and deleted must not be suggested again by the algorithms, that must hence learn
            from their mistake. The current implementation is quite flawed, I've just thrown it there to remember
            something ought to be done about it.
        Data display (dev):
            A few methods are offered to ease the display of data related to stones finding. For example things
            about intersections benefit from some automation.

    Attributes:
        goban_img: ndarray
            The current goban image.
        canonical_shape: (int, int)
            the shape of the image after transform (extraction of goban pixels only)
        _posgrid: PosGrid
            Holds the positions of each intersection of the grid.
        mask_cache: ndarray
            A mask that help keeping a disk around each intersection and blacking out other pixels.
        zone_area: int
            The area of a zone (non-zero pixels of the mask).
        intersections: ndarray
            Cache for the result of self.find_intersections().

        bg_model: BackgroundSubtractor
            Responsible for background learning and segmentation.
        bg_init_frames: int
            The number of frames that should be reserved for background learning at startup.

        corrections: Queue
            Corrections the user has recently made (fixing stones detection) that have not yet been learnt from.
        saved_bg: ndarray
            A background image used in the "user corrections" learning process.
        deleted: dict
            Locations under "deletion watch". keys: the locations, values: the number of samples left to do.
            Note: each location is a tuple indicating an intersection row and column, in numpy coordinates frame.
        nb_del_samples: int
            The number of times a zone should be sampled when it has been deleted by user. For example, after the
            user has deleted a wrongly detected stone, compute the mean of the pixel area around that intersection
            over 50 frames.
    """

    def __init__(self, vmanager, learn_bg=True):
        """
        Args:
            vmanager: VManager
                Used to get references to BoardFinder and Controller, as well as image queue, and frame reading.
            learn_bg: bool
                Set to True to create and maintain a background model, which enables self.get_foreground().

        """
        super().__init__(vmanager)
        self.goban_img = None
        self.canonical_shape = (cvconf.canonical_size, cvconf.canonical_size)
        self._posgrid = PosGrid(cvconf.canonical_size)
        self.mask_cache = None
        self.zone_area = None
        self.intersections = None

        # background-related attributes
        if learn_bg:
            self.bg_model = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            self.bg_init_frames = 50

        # (quite primal) "learning" attributes. see self._learn()
        self.corrections = queue.Queue(correc_size)
        self.saved_bg = np.zeros(self.canonical_shape + (3,), dtype=np.float32)
        self.deleted = {}
        self.nb_del_samples = 50

    def _doframe(self, frame):
        """ Abstract implementation of frame processing by a stones finder.

        Compute the image transform to isolate and "straighten up" the Goban's area if the board has been found.
        Then call the default routines, and finally delegate the actual detection part to self._find() which is
        abstract.

        Args:
            frame: ndarray
                The image to process (where to find stones).
        """
        transform = None
        self.intersections = None  # reset cache
        if self.vmanager.board_finder is not None:
            transform = self.vmanager.board_finder.mtx
        if transform is not None:
            try:
                self.goban_img = cv2.warpPerspective(frame, transform, self.canonical_shape)
            except cv2.error as e:
                print("frame:", frame, sep="\n")
                print("transform:", transform, sep="\n")
                raise e
            self._learn_bg()
            self._learn()
            self._find(self.goban_img)
        else:
            if 1 < time.time() - self.last_shown:
                black = np.zeros(self.canonical_shape, dtype=np.uint8)
                imgutil.draw_str(black, "NO BOARD LOCATION AVAILABLE", int(black.shape[0] / 2 - 110), int(black.shape[1] / 2))
                self._show(black)

    def ready_to_read(self):
        """ Indicate whether we are ready to process frames (eg. we're not if the board location is not known).
        """
        try:
            return super().ready_to_read() and self.vmanager.board_finder.mtx is not None
        except AttributeError:
            return False

    def _find(self, goban_img):
        """ Abstract method. Detect stones in the (already) canonical image of the goban.

        Args:
            goban_img: ndarray
                The already straightened up Goban image. It is expected to only contain the Goban pixels.
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _learn_bg(self):
        """ Apply the last frame read to the background model, and save the resulting foreground.
        """
        if hasattr(self, "bg_model"):
            learning = 0.01 if self.total_f_processed < self.bg_init_frames else 0.005
            self._fg = self.bg_model.apply(self.goban_img, learningRate=learning)

    def _learn(self):
        """ Process user corrections queue to try to learn from it. Partial implementation.

        This partial implementation only supports a basic reaction to deletion. The idea is to try to "remember"
        what each False positive (wrongly detected stone) zone looked like when the user makes the correction (delete),
        so that new suggestions by algorithms at this location rise an error if the pixels haven't changed much in
        this zone.

        Other cases will raise an Exception for now, to show they exist but not reaction has been implemented.
        """
        # step one: process new user inputs
        unprocessed = []
        try:
            while True:
                err, exp = self.corrections.get_nowait()
                if exp is None:
                    # a deletion has occurred, need to secure the "emptied" location
                    self.deleted[(err.y, err.x)] = self.nb_del_samples  # Move.x and Move.y are in openCV coord system
                elif err is not None and (err.x, err.y) != (exp.x, exp.y):
                    # a stone has been moved, need to secure the "emptied" location
                    self.deleted[(err.y, err.x)] = self.nb_del_samples
                else:
                    # missed a stone (not that bad), leave it to subclasses to figure if they want to use this info.
                    unprocessed.append((err, exp))
        except queue.Empty:
            pass

        # step 2: sample all locations that still need it
        for (r, c), nb_left in self.deleted.items():
            if nb_left:
                # only sample "calm" frames if background segmentation is on
                try:
                    fg = self.get_foreground()
                except ValueError:
                    fg = None
                x0, y0, x1, y1 = self.getrect(r, c)
                if fg is None or np.sum(fg[x0:x1, y0:y1] < 0.1 * (x1-x0) * (y1-y0)):
                    # noinspection PyTypeChecker
                    self.saved_bg[x0:x1, y0:y1] += self.goban_img[x0:x1, y0:y1] / self.nb_del_samples
                    self.deleted[(r, c)] = nb_left - 1

        # bonus step: forcibly indicate any ignored input (the user added stones, or changed the color of stones)
        if 0 < len(unprocessed):
            raise camkifu.core.CorrectionWarning(unprocessed, message="Unhandled corrections")

    def _check_dels(self, r: int, c: int):
        """ Check that the given intersection has not been deleted by user recently.

        Args:
            r: int
            c: int
                The row and column of the intersection to check, in numpy coord system.
        """
        try:
            nb_samples_left = self.deleted[(r, c)]
        except KeyError:
            return  # all good, this location is not under deletion watch
        if 0 == nb_samples_left:  # only check when sampling has completed
            x0, y0, x1, y1 = self.getrect(r, c)
            diff = self.saved_bg[x0:x1, y0:y1] - self.goban_img[x0:x1, y0:y1]
            if np.sum(np.absolute(diff)) / (diff.shape[0] * diff.shape[1]) < 40:
                raise camkifu.core.DeletedError(((r, c),), "The zone has not changed enough since last deletion.")
            else:
                # the area has changed, alleviate ban.
                print("previously user-deleted location: {} now unlocked".format((r, c)))
                del self.deleted[(r, c)]
        else:
            raise camkifu.core.DeletedError(((r, c),), "The zone has been marked as deleted too recently.")

    def _window_name(self):
        return "camkifu.stone.stonesfinder.StonesFinder"

    def suggest(self, color, r: int, c: int, doprint=True):
        """ Indicate to the controller the add of a new stone on the goban. May be processed asynchronously.

        Args:
            color: B or W
                The color of the new stone to add.
            r: int
            c: int
                The row and column of the intersection where to add the stone, in numpy coord system.
            doprint: bool
                Whether or not to print the move in the console.
        """
        self._check_dels(r, c)
        move = golib.model.Move('np', ctuple=(color, r, c))
        if doprint:
            print(move)
        self.vmanager.controller.pipe("append", move)

    def remove(self, x, y):
        """ Indicate to the controller the removal of a stone from the goban. May be processed asynchronously.

        Although allowing automated removal of stones doesn't seem to be a very safe idea given the current
        robustness of stones finders, here's an implementation.

        Args:
            r: int
            c: int
                The row and column of the intersection where to remove the stone, in numpy coord system.
        """
        assert not self.is_empty(x, y), "Can't remove stone from empty intersection."
        move = golib.model.Move('np', ("", x, y))
        print("delete {}".format(move))
        self.vmanager.controller.pipe("delete", move.x, move.y)

    def bulk_update(self, tuples):
        """ Indicate to the controller a series of updates have happened on the goban. May be processed asynchronously.

        Note: if a move points to an already occupied location but with a different color, the previous stone is
        removed, then the new one is added. The color can't simply be changed due to consistency issues, notably
        some previous kills may have to be invalidated and the history reworked accordingly.

        Args:
            tuples: [ (color1, r1, c1), (color2, r2, c2), ... ]
                The list of update moves. A move with color E is interpreted as a removed stone.
                r and c, the intersection row and column, are in numpy coord system.
        """
        moves = []
        del_errors = []
        for color, x, y in tuples:
            if color is E and not self.is_empty(x, y):
                    moves.append(golib.model.Move('np', (color, x, y)))
            elif color in (B, W):
                if not self.is_empty(x, y):
                    existing_mv = self.vmanager.controller.locate(y, x)
                    if color is not existing_mv.color:  # if existing_mv is None, better to crash now, so no None check
                        # delete current stone to be able to put the other color
                        moves.append(golib.model.Move('np', (E, x, y)))
                    else:
                        continue  # already up to date, go to next iteration
                try:
                    self._check_dels(x, y)
                    moves.append(golib.model.Move('np', (color, x, y)))
                except camkifu.core.DeletedError as de:
                    del_errors.append(de)
        if len(moves):
            self.vmanager.controller.pipe("bulk", moves)
        if len(del_errors):
            msg = "Bulk_update:warning: All non-conflicting locations have been sent."
            raise camkifu.core.DeletedError(del_errors, message=msg)

    def corrected(self, err_move: golib.model.Move, exp_move: golib.model.Move) -> None:
        """ Indicate that a correction has been made by the user to a stone presence/location on the Goban.

        Actual processing of corrections is expected to be done asynchronously on the StonesFinder thread.
        See _learn().

        Args:
            err_move: Move
                A move that has wrongly been detected by algorithms. It has been removed by user.
            exp_move: Move
                A move that has been missed by detection algorithms. It has been added by user.
        Note: a relocation will for example consist of the providing of an err_move together with an exp_move.
        """
        try:
            self.corrections.put_nowait((err_move, exp_move))
        except queue.Full:
            print("Corrections queue full (%s), ignoring %s -> %s" % (correc_size, str(err_move), str(exp_move)))

    def is_empty(self, r: int, c: int) -> bool:
        """ Return True if the provided goban position is empty (color is E). Synchronized method.

        Args:
            x: int
            y: int
                The row and column of the intersection to check, in numpy coord system.
        """
        return self.vmanager.controller.is_empty_blocking(c, r)

    def _empties(self) -> (int, int):
        """ Yield the unoccupied positions of the goban in naive order.
        Note: this implementation allows for the positions to be updated concurrently to yielding.

        Yields:
            r, c: int, int
                The row and column of the next empty intersection, in the numpy coordinates system.
        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.is_empty_blocking(x, y):
                    yield y, x

    def _empties_spiral(self) -> (int, int):
        """ Yield the unoccupied positions of the goban along an inward spiral.

        Aims to help detect hand / arm appearance faster by analysing outer border(s) first.
        Note: this implementation allows for the positions to be updated concurrently to yielding.

        Yields:
            r, c: int, int
                The row and column of the next empty intersection, in the numpy coordinates system.
        """
        inset = 0
        while inset <= gsize / 2:
            for x, y in self._empties_border(inset):
                yield x, y
            inset += 1

    def _empties_border(self, inset):
        """ Yield unoccupied positions of the goban along a defined square.

        Args:
            inset: int
                The inner margin defining the square along with to check for empty intersections.
                Eg. if inset == 2, yield the empty positions on the third line of the Goban.

        Yields:
            r, c: int, int
                The row and column of the next empty intersection, in the numpy coordinates system.
        """
        y = inset
        for x in range(inset, gsize - inset):
            if self.vmanager.controller.is_empty_blocking(x, y):
                yield y, x

        x = gsize - inset - 1
        for y in range(inset + 1, gsize - inset):
            if self.vmanager.controller.is_empty_blocking(x, y):
                yield y, x

        y = gsize - inset - 1
        for x in range(gsize - inset - 2, inset - 1, -1):  # reverse just to have a nice order. not actually useful
            if self.vmanager.controller.is_empty_blocking(x, y):
                yield y, x

        x = inset
        for y in range(gsize - inset - 2, inset, -1):
            if self.vmanager.controller.is_empty_blocking(x, y):
                yield y, x

    def getrect(self, r: int, c: int, cursor: float=1.0) -> (int, int, int, int):
        """ Return the rectangle of pixels around the provided goban intersection (r, c).

        This method relies on self._posgrid.mtx to get the coordinates of the intersections (so they apply in an
        image of the same size as the canonical frame).

        Args:
            r: int
            c: int
                The intersection row and column, in numpy coordinates system.
            cursor: float
                Has sense in the interval ]0, 2[ as follows:
                0 -> the zone is restricted to the (r, c) point.
                2 -> the zone is delimited by the rectangle (r-1, c-1), (r+1, c+1).
                1 -> the zone is a rectangle of "intuitive" size, halfway between the '0' and '2' cases.

        Returns x0, y0, x1, y1: int, int, int, int
            The rectangle coordinates, as diagonal points.
        """
        assert isinstance(cursor, float)
        p = self._posgrid.mtx[r][c]
        pbefore = self._posgrid.mtx[r - 1][c - 1].copy()
        pafter = self._posgrid.mtx[min(r + 1, gsize - 1)][min(c + 1, gsize - 1)].copy()
        if r == 0:
            pbefore[0] = -p[0]
        elif r == gsize - 1:
            pafter[0] = 2 * self._posgrid.size - p[0] - 2
        if c == 0:
            pbefore[1] = -p[1]
        elif c == gsize - 1:
            pafter[1] = 2 * self._posgrid.size - p[1] - 2

        # determine start and end point of the rectangle
        w = cursor / 2
        x0 = max(0, int(w * pbefore[0] + (1 - w) * p[0]))
        y0 = max(0, int(w * pbefore[1] + (1 - w) * p[1]))
        x1 = min(self._posgrid.size, int((1 - w) * p[0] + w * pafter[0]))
        y1 = min(self._posgrid.size, int((1 - w) * p[1] + w * pafter[1]))
        return x0, y0, x1, y1

    def getmask(self, depth=1) -> np.ndarray:
        """ Compute a boolean mask that isolates a circle around each goban intersection.

        Multiply a Goban frame by this mask to zero-out anything outside the circles. This mask is shaped
        in "cvconf.canonical_size".

         Args:
            depth: int
                The size of the third dimension (== number of color channels).

        Returns mask_cache: ndarray
            The boolean mask.
        """
        if self.mask_cache is None or depth != (1 if len(self.mask_cache.shape) == 2 else self.mask_cache.shape[2]):
            print("initializing stones mask")
            shape = self.canonical_shape
            mask = np.empty(shape, dtype=np.uint8)  # without depth
            for row in range(gsize):
                for col in range(gsize):
                    x0, y0, x1, y1 = self.getrect(row, col)
                    zone = mask[x0:x1, y0:y1]
                    a = zone.shape[0] / 2
                    b = zone.shape[1] / 2
                    r = min(a, b)
                    y, x = np.ogrid[-a:zone.shape[0] - a, -b: zone.shape[1] - b]
                    zmask = x * x + y * y <= r * r
                    mask[x0:x1, y0:y1] = zmask
            if 1 < depth:
                shape += (depth, )
            self.mask_cache = np.empty(shape)

            # duplicate mask to match image depth
            if len(shape) == 3:
                for i in range(self.mask_cache.shape[2]):
                    self.mask_cache[:, :, i] = mask
            else:
                self.mask_cache[:] = mask

            # store the area of one zone for normalizing purposes
            x0, y0, x1, y1 = self.getrect(0, 0)
            self.zone_area = np.sum(mask[x0:x1, y0:y1])
        return self.mask_cache

    def get_stones(self):
        """
        Returns stones: ndarray
            A copy of the current goban state, in the numpy coordinates system.
        """
        return self.vmanager.controller.get_stones()

    def get_foreground(self):
        """ Return the foreground of the last frame read if background segmentation is on, otherwise raise ValueError.

        Returns fg: ndarray
            The foreground mask.
        """
        if hasattr(self, "_fg"):
            current, target = self.total_f_processed, self.bg_init_frames
            if current < target:
                print("Warning : background model still initializing ({} / {})".format(current,  target))
            return self._fg
        else:
            raise ValueError("This StonesFinder doesn't seem to be segmenting background. See self.__init__()")

    def find_intersections(self, img: np.ndarray, canvas: np.ndarray=None) -> np.ndarray:
        """ Find which intersections are likely to be empty based on lines detection.

        The search is based on hough lines detection: if good lines are found inside the intersection zone, it is
        very unlikely that a stone would be present. This method is also used to update the intersections coordinates
        when "crosses" are found. By "cross" read at least one horizontal and one vertical line.

        Args:
            img: ndarray
                The Goban image (in the canonical frame).
            canvas: ndarray
                An optional image onto which draw some results.

        Returns grid: ndarray
            A matrix containing the intersections positions (as per PosGrid). The positions where at least one line
            has been found are indicated by negation (* -1). Furthermore, on top of being negated, for each "cross"
            found the corresponding intersection coordinates are updated.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh, _ = cv2.threshold(gray, 1, 1, cv2.THRESH_OTSU)
        canny = cv2.Canny(gray, thresh/2, thresh)
        # noinspection PyNoneFunctionAssignment
        grid = self._posgrid.mtx.copy()
        for r in range(gsize):
            for c in range(gsize):
                x0, y0, x1, y1 = self.getrect(r, c)
                zone = canny[x0:x1, y0:y1]
                min_side = min(zone.shape[0], zone.shape[1])
                thresh = int(min_side * 3 / 4)
                min_len = int(min_side * 2 / 3)
                lines = cv2.HoughLinesP(zone, 1, math.pi / 180, threshold=thresh, maxLineGap=0, minLineLength=min_len)
                if lines is not None and 0 < len(lines):
                    if canvas is not None:
                        imgutil.draw_lines(canvas[x0:x1, y0:y1], [line[0] for line in lines])
                    update_grid(lines, (x0, y0, x1, y1), grid[r][c])
        return grid

    def get_intersections(self, img, display=False) -> np.ndarray:
        """ A cached wrapper of self.find_intersections().

        See self.find_intersections().

        Args:
            img: ndarray
                The Goban image (in the canonical frame).
            display: bool
                True to display the empty intersections found on a new image.

        Returns grid: ndarray
            As per self.find_intersections().
        """
        if self.intersections is None:
            if display:
                canvas = img.copy()
            else:
                canvas = None
            self.intersections = self.find_intersections(img, canvas=canvas)
            self._posgrid.learn(np.absolute(self.intersections))
            if display:
                self.display_intersections(self.intersections, canvas)
        return self.intersections

    def stone_radius(self) -> float:
        """ The approximation of the radius (in pixels) of a stone in the canonical frame.
        Simple implementation based on self._posgrid.size.

        Returns radius: float
        """
        return self._posgrid.size / gsize / 2

    def stone_boxarea_bounds(self) -> (float, float):
        """ Compute the minimum and maximum areas of a contour's bounding box, that may be candidate to be a stone.
        These bounds are based on the default estimated box area for a stone: (2 * self.stone_radius()) ** 2.

        Return min_area, max_area: float, float
        """
        radius = self.stone_radius()
        min_area = (4 / 3 * radius) ** 2
        max_area = (3 * radius) ** 2
        return min_area, max_area

    def check_against(self, stones: np.ndarray, reference: np.ndarray=None, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ Check that the newly found stones array is coherent with the already existing reference array.
        Both new and old arrays can be subregions of the Goban, as long as they have the same shape.

        Args:
            stones: ndarray
                The newly found stones, in other words the result to check.
            reference: ndarray
                The already found stones, in other words the current status of the Goban.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Although not used explicitly, allowing for keyword args enables multiple check methods to be called
                indifferently. See self.verify().

        Return status: int
            -1, 0, or 1 if the check is respectively refused, undetermined, or passed.
        """
        refs = 0  # number of stones found in reference
        matches = 0  # number of matches with the references (non-empty only)
        for r in range(rs, re):
            for c in range(cs, ce):
                if reference[r, c] in (B, W):
                    refs += 1
                    if stones[r, c] is reference[r, c]:
                        matches += 1
        if 4 < refs:  # check against at least 4 stones
            if 0.81 < matches / refs:  # start allowing errors if more than 5 stones have been put
                return 1  # passed
            else:
                return -1  # refused
        return 0  # undetermined

    def check_lines(self, stones: np.ndarray, img: np.ndarray=None, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ Check that the provided stones array is coherent with lines detection in the image.

        No line should be found in zones where a stone has been detected. A match is counted for the zone if it is
        empty (E) and at least one line has also been detected in that zone.

        Args:
            stones: ndarray
                The newly found stones, in other words the result to check.
            img: ndarray
                The Goban image where to find the lines.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Allowing for keyword args enables multiple check methods to be called indifferently. See self.verify().

        Return status: int
            1 if enough matches have been found,
           -1 if too many mismatches have been found,
            0 if data isn't sufficient to judge (eg. too few lines found, potentially caused by too few empty positions)
        """
        grid = self.get_intersections(img)
        lines = 0  # the number of intersections where lines have been detected
        matches = 0  # the number of empty intersections where lines have been detected
        for r in range(rs, re):
            for c in range(cs, ce):
                if sum(grid[r, c]) < 0:
                    lines += 1
                    if stones[r, c] is E:
                        matches += 1
        if 4 < lines:
            if 0.9 < matches / lines:
                return 1  # passed
            else:
                return -1  # refused
        return 0  # undetermined

    def check_thickness(self, stones: np.ndarray, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ Check that the provided stones array doesn't contain "big chunks" that wouldn't make sense in a game of Go.

        Args:
            stones: ndarray
                The newly found stones, in other words the result to check.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Allowing for keyword args enables multiple check methods to be called indifferently. See self.verify().

        Return status: int
            -1, 0, or 1 if the check is respectively refused, undetermined, or passed.
        """
        for color in (B, W):
            avatar = np.vectorize(lambda x: 1 if x is color else 0)(stones[rs:re, cs:ce].flatten())
            # diagonal moves cost as much as side moves
            dist = cv2.distanceTransform(avatar.reshape((re-rs, ce-cs)).astype(np.uint8), cv2.DIST_C, 3)
            if 2 < np.max(dist):
                # a stone surrounded by a 2-stones-thick wall of its own color is most likely not Go
                return -1
        return 0

    def check_flow(self, stones: np.ndarray, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ Check that newly added stones colors match expectations.

        If multiple new stones : there should be at most one more stone of a color than the other color. The use of
        this check may prevent algorithms from catching up with previous failures to detect a certain color of stones,
        since they won't be allowed to add multiple stones of the same color when they finally see them. In other
        words this check is probably not such a good idea (As of 14/02/15).

        Args:
            stones: ndarray
                The newly found stones, in other words the result to check.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Allowing for keyword args enables multiple check methods to be called indifferently. See self.verify().

        Return status: int
            -1, 0, or 1 if the check is respectively refused, undetermined, or passed.
        """
        moves = []  # newly added moves
        for r in range(rs, re):
            for c in range(cs, ce):
                if self.is_empty(r, c) and stones[r, c] is not E:
                    moves.append(stones[r, c])
        # the total color counts in new moves should differ by at most 1
        else:
            diff = 0
            for mv in moves:
                diff += 1 if mv is B else -1
            # can't really confirm, but at least nothing seems wrong
            return 0 if abs(diff) <= 1 else -1

    def first_line_lonelies(self, stones: np.ndarray, reference: np.ndarray=None, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ Find the stones on the first line that have no neighbour in a 2-lines-thick square around them.

        Args:
            stones: ndarray
                The newly found stones, in other words the result to check.
            reference:
                The already found stones. Required because 'stones' may only contain newly detected stones.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Allowing for keyword args enables multiple check methods to be called indifferently. See self.verify().

        Returns lonelies: list
            The lonely stones coordinates (r, c), in numpy coord system.
        """
        pos = set()
        for r in (rs, re):
            if r in (0, gsize - 1):
                for c in range(cs, ce):
                    pos.add((r, c))
        for c in (cs, ce):
            if c in (0, gsize - 1):
                for r in range(rs, re):
                    pos.add((r, c))
        lonelies = []
        for (r, c) in pos:
            if stones[r, c] in (B, W):
                alone = True
                for x, y in imgutil.around(r, c, 2, xmin=0, xmax=gsize, ymin=0, ymax=gsize):
                    if reference[x, y] in (B, W) or stones[x, y] in (B, W):
                        alone = False
                        break
                if alone:
                    lonelies.append((r, c))
        return lonelies

    def _drawgrid(self, img: np.ndarray):
        """ Draw a circle around each intersection of the goban, as they are currently estimated.

        Args:
            img: ndarray
                The image on which to draw.
        """
        if self._posgrid is not None:
            for i in range(19):
                for j in range(19):
                    p = self._posgrid.mtx[i][j]
                    cv2.circle(img, (p[1], p[0]), 5, (255, 255, 0))

    def _drawvalues(self, img, values):
        """ Display one value per goban position on the image. Obviously values will soon overlap if they are longish.

        Args:
            img: ndarray
                The image on which to draw.
            values: 2D iterable
                The values to draw. Must be of shape (gsize, gsize), in other words one value per intersection.
        """
        for row in range(gsize):
            for col in range(gsize):
                x, y = self._posgrid.mtx[row, col]
                imgutil.draw_str(img, str(values[row, col]), x - 10, y + 2)

    def draw_stones(self, stones: np.ndarray, canvas: np.ndarray=None):
        """ Dev method to display an array of stones in an image.

        It is a simpler alternative than suggesting them to the goban, since there's no game logic involved
        (easier to update on the flight to get an idea of what's going on).

        Args:
            stones: ndarray
                The stones to display. Expected to be of shape (gsize, gsize).
            canvas: ndarray
                The image on which to draw the stones. If not provided, a blank (actually, brown) canvas is created.

        Returns canvas: ndarray
            The updated image.
        """
        if canvas is None:
            canvas = np.zeros((self._posgrid.size, self._posgrid.size, 3), dtype=np.uint8)
            canvas[:] = (65, 100, 128)
        for x in range(gsize):
            for y in range(gsize):
                if stones[x][y] is B:
                    color = (0, 0, 0)
                elif stones[x][y] is W:
                    color = (255, 255, 255)
                else:
                    continue
                p = self._posgrid.mtx[x][y]
                cv2.circle(canvas, (p[1], p[0]), 10, color, thickness=-1)
        self._drawgrid(canvas)
        return canvas

    def display_intersections(self, grid, img):
        """ Dev method showing how the intersection analysis is doing, in a dedicated window.

        The intersections that have been detected as empty have a circle, and those that have been updated as well
        have an additional black border. Also, there's been some care given to line and column visual differentiation,
        hope it makes sens :)

        Args:
            grid: ndarray
                The intersections location grid, as returned by self.find_intersections().
            img: ndarray
                The image on which to draw.
        """
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                inter = grid[i][j]
                dif = self._posgrid.mtx[i][j] + inter
                if 0 < abs(dif[0]) + abs(dif[1]):
                    # convert back to opencv coords frame
                    cv2.circle(img, (-inter[1], -inter[0]), 5, color=(0, 0, 0), thickness=6)
                cv2.circle(img, (-inter[1], -inter[0]), 5, color=self.get_display_color(i, j), thickness=2)
        self._show(img, "{} - Intersections".format(self._window_name()))

    @staticmethod
    def get_display_color(i, j):
        """ Compute a color for the i, j intersection. To be used in display_intersections()
        """
        factor = 0.5 if i % 2 else 1
        blue = 40 / factor
        green = (230 if j % 2 else 10) * factor
        red = (10 if j % 2 else 200) * factor
        return blue, green, red

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_frequ=2):
        """ Override to take control of the location of the window of this stonesfinder.
        """
        if loc is None:
            from camkifu.config.cvconf import sf_loc
            loc = sf_loc
        super()._show(img, name, latency, thread, loc=loc, max_frequ=max_frequ)

    def display_bg_sampling(self, shape):
        """ Display a "message" image of the provided shape, indicating the background sampling is running.
        """
        black = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        message = "BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames)
        imgutil.draw_str(black, message)
        self._show(black)


def update_grid(lines, box, result_slot):
    """ Analyse the lines found for the intersection defined by 'box', to determine if it is empty.

    Negative result values indicate that the zone (probably) contains no stone. The negative values may have been
    updated as well with new coordinates if a "cross" has benn found.

    Precisely:
        - if there's only one valid line, or too many lines, mark the zone as empty (negate result_slot).
        - if there is a decent (1 < x < 5) number of lines, compute the center of mass of their intersections
          that are located inside the zone (including a safety margin). then update result_slot accordingly
          with the new coordinates (negated).

    Args:
        lines: iterable
            As returned by cv2.HoughLinesP().
        box: (int, int, int, int)
            (x0, y0, x1, y1) - The rectangle delimiting the intersection in which the lines have been found.
        result_slot: ndarray
            The part of the locations grid corresponding to the current intersection.
    """
    margin = min(box[2] - box[0], box[3] - box[1]) / 7
    # step one: only retain lines that are either vertical or horizontal enough, and centered
    segments = []
    for line in lines:
        seg = imgutil.Segment(line[0])
        if 0.995 < abs(math.cos(seg.theta)):  # horizontal check + margin respect
            p = (box[0] + box[2]) / 2, (seg.p1()[0] + seg.p2()[0]) / 2 + box[1]
            if not imgutil.within_margin(p, box, margin):
                continue
        elif 0.995 < abs(math.sin(seg.theta)):  # vertical check + margin respect
            p = (seg.p1()[1] + seg.p2()[1]) / 2 + box[0], (box[3] + box[1]) / 2  # numpy coordinates
            if not imgutil.within_margin(p, box, margin):
                continue
        else:
            continue
        seg.offset = box[1], box[0]  # swap "points" to opencv coord
        segments.append(seg)
    # step two, analyse filtered lines
    if len(segments):
        # if at least one line present, indicates that the intersection is occupied
        result_slot[:] *= -1
        # pass
    # then if there's a decent amount of lines, try to refine intersection location
    if 1 < len(segments) < 5:
        x_sum = 0
        y_sum = 0
        count = 0
        for seg1 in segments:
            for seg2 in segments:
                if seg1 is not seg2:
                    p = seg1.intersection(seg2)
                    if p is not None:
                        p = p[1] + box[0], p[0] + box[1]  # add offset to match global image coordinates
                        if imgutil.within_margin(p, box, margin):
                            x_sum += p[0]
                            y_sum += p[1]
                            count += 1
        if 0 < count:
            result_slot[0] = - x_sum / count
            result_slot[1] = - y_sum / count


class PosGrid:
    """ Store the location of each intersection of the goban, in numpy coordinates system.

    Attributes:
        size: int
            The length in pixels of one side of the goban canonical frame (supposed to be a square for now).
        mtx: ndarray
            The pixel position of each intersection of the goban.
        adjust_vect: [x, y]
            An accumulator of adjustments that should be made to the overall grid position.
        adjust_contribs: int
            The number of accumulations applied to 'adjust_vect' so far (since last reset).
    """

    def __init__(self, size):
        self.size = size
        self.mtx = np.zeros((gsize, gsize, 2), dtype=np.int16)
        self.adjust_vect = np.zeros(2, dtype=np.float32)
        self.adjust_contribs = 0

        # initialize grid with default values
        start = size / gsize / 2
        end = size - start
        hull = [(start, start), (end, start), (end, end), (start, end)]
        for i in range(gsize):
            xup = (hull[0][0] * (gsize - 1 - i) + hull[1][0] * i) / (gsize - 1)
            xdown = (hull[3][0] * (gsize - 1 - i) + hull[2][0] * i) / (gsize - 1)
            for j in range(gsize):
                self.mtx[i][j][0] = (xup * (gsize - 1 - j) + xdown * j) / (gsize - 1)
                yleft = (hull[0][1] * (gsize - 1 - j) + hull[3][1] * j) / (gsize - 1)
                yright = (hull[1][1] * (gsize - 1 - j) + hull[2][1] * j) / (gsize - 1)
                self.mtx[i][j][1] = (yleft * (gsize - 1 - i) + yright * i) / (gsize - 1)

    def closest_intersection(self, point):
        """ Find the closest intersection from the given (x,y) point of the canonical image (goban image).

        Args:
            point: (int, int)
                A location in the image, in numpy coordinates system.

        Returns intersection: (int, int)
            The closest goban row and column, in numpy coordinates system.
        """
        # a smarter version would use the fact that self.learn() shifts the whole grid to store and apply the offset
        # over time, and then return the hook initialization as below. yet that's a more generic implementation.
        hook = None
        target = (int(point[0] / self.size * gsize), int(point[1] / self.size * gsize), self.size)
        while target != hook:
            hook = target
            # not neat: if more than one iteration is performed, 3 to 5 distances are computed twice each time !
            # but premature optimization is the root of evil :)
            for i in range(-1, 2):
                x = hook[0] + i
                for j in range(-1, 2):
                    y = hook[1] + j
                    try:
                        dist = sum(np.absolute(self.mtx[x][y] - point))
                        if dist < target[2]:
                            target = (x, y, dist)
                    except IndexError:
                        pass
        # at least one point must have been calculated successfully. better assert that since exceptions are silenced
        assert target[2] < self.size
        return target[0], target[1]

    def learn(self, grid, rate=0.2):
        """ Update the current intersections locations with the provided grid.

        Compute the mean diff vector between the provided grid and the current mtx. Then shift all the positions of
        the current grid by this vector.

        Args:
            grid: ndarray
                An updated version of self.mtx
            rate: float
                A factor used to scale the adjustment vector before it is applied.
        """
        assert 0 < rate <= 1  # if 0, why call ?
        diff = grid - self.mtx
        if np.min(diff) < -200:
            raise ValueError("Provided grid seems too far from original, at least for one point.")
        # the number of intersections that are contributing to adjust the grid
        contributors = np.count_nonzero(np.absolute(diff[:, :, 0]) + np.absolute(diff[:, :, 1]))
        if 0 < contributors:
            vect = np.sum(diff, axis=(0, 1), dtype=np.float32)
            vect /= contributors
            self.adjust_vect *= (1.0 - rate)
            self.adjust_vect += vect * rate
            self.adjust_contribs += contributors
            if 20 < self.adjust_contribs:
                print("Grid adjust vector : {}".format(self.adjust_vect))
                self.mtx += self.adjust_vect.astype(np.int16)
                self.adjust_vect[:] = 0
                self.adjust_contribs = 0
