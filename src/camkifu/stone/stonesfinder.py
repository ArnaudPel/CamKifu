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


__author__ = 'Arnaud Peloquin'


correc_size = 10


class StonesFinder(camkifu.core.VidProcessor):
    """
    Abstract class providing a base structure for stones-finding processes.
    It relies on the providing of a transform matrix to extract only the goban pixels from the global frame.

    """

    def __init__(self, vmanager, learn_bg=True):
        """
        learn_bg -- set to True to create and maintain a background model, which enables self.get_foreground().

        """
        super().__init__(vmanager)
        self.goban_img = None  # the current goban image
        self.canonical_shape = (cvconf.canonical_size, cvconf.canonical_size)
        self._posgrid = PosGrid(cvconf.canonical_size)
        self.mask_cache = None
        self.zone_area = None  # the area of a zone # (non-zero pixels of the mask)
        self.intersections = None  # cache for the result of self.find_intersections()

        # background-related attributes
        if learn_bg:
            self.bg_model = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            self.bg_init_frames = 50

        # (quite primal) "learning" attributes. see self._learn()
        self.corrections = queue.Queue(correc_size)
        self.saved_bg = np.zeros(self.canonical_shape + (3,), dtype=np.float32)
        self.deleted = {}  # locations under "deletion watch". keys: the locations, values: the number of samples to do
        self.nb_del_samples = 5

    def _doframe(self, frame):
        transform = None
        self.intersections = None  # reset cache
        if self.vmanager.board_finder is not None:
            transform = self.vmanager.board_finder.mtx
        if transform is not None:
            try:
                self.goban_img = cv2.warpPerspective(frame, transform, self.canonical_shape)
            except cv2.error:
                print("frame:", frame, sep="\n")
                print("transform:", transform, sep="\n")
            self._learn_bg()
            self._learn()
            self._find(self.goban_img)
        else:
            if 1 < time.time() - self.last_shown:
                black = np.zeros(self.canonical_shape, dtype=np.uint8)
                imgutil.draw_str(black, "NO BOARD LOCATION AVAILABLE", int(black.shape[0] / 2 - 110), int(black.shape[1] / 2))
                self._show(black)

    def ready_to_read(self):
        """
        Don't read frames if the board location is not known.

        """
        try:
            return super().ready_to_read() and self.vmanager.board_finder.mtx is not None
        except AttributeError:
            return False

    def _find(self, goban_img):
        """
        Detect stones in the (already) canonical image of the goban.

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _learn_bg(self):
        """
        Apply img to the background learning structure, and save the resulting foreground.

        """
        if hasattr(self, "bg_model"):
            learning = 0.01 if self.total_f_processed < self.bg_init_frames else 0.005
            self._fg = self.bg_model.apply(self.goban_img, learningRate=learning)

    def _learn(self):
        """
        Process user corrections queue (partial implementation).

        This partial implementation only supports a basic reaction to deletion. The idea is to try to "remember"
        what each False positive (wrongly detected stone) zone look like when the user makes the correction (delete),
        so that new suggestions by algorithms at this location rise an error as long as the pixels haven't changed
        much in this zone.

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
                # only sample "calm" frames
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
        try:
            nb_samples_left = self.deleted[(r, c)]
        except KeyError:
            return  # this location is not under deletion watch
        if 0 == nb_samples_left:  # only check when sampling has completed
            x0, y0, x1, y1 = self.getrect(r, c)
            diff = self.saved_bg[x0:x1, y0:y1] - self.goban_img[x0:x1, y0:y1]
            if np.sum(np.absolute(diff)) / (diff.shape[0] * diff.shape[1]) < 40:
                raise camkifu.core.DeletedError("The zone has not changed enough since last deletion")
            else:
                # the area has changed, alleviate ban.
                print("previously user-deleted location: {} now unlocked".format((r, c)))
                del self.deleted[(r, c)]
        else:
            raise camkifu.core.DeletedError("The zone has been marked as deleted only a few frames ago")

    def _window_name(self):
        return "camkifu.stone.stonesfinder.StonesFinder"

    def suggest(self, color, x: int, y: int, doprint=True):
        """
        Suggest the add of a new stone to the goban.
        -- color : in (E, B, W)
        -- x, y : the coordinates
        -- ctype : the type of coordinates frame (see Move._interpret()), defaults to 'np' (numpy)

        """
        self._check_dels(x, y)
        move = golib.model.Move('np', ctuple=(color, x, y))
        if doprint:
            print(move)
        self.vmanager.controller.pipe("append", move)

    def remove(self, x, y):
        """
        Although allowing automated removal of stones doesn't seem to be a very safe idea given the current
        robustness of stones finders, here's an implementation.

        x, y -- the location in numpy coordinates frame.
        ctype -- the type of coordinates frame (see Move._interpret()), defaults to 'np' (numpy)

        """
        assert not self.is_empty(x, y), "Can't remove stone from empty intersection."
        move = golib.model.Move('np', ("", x, y))
        print("delete {}".format(move))
        self.vmanager.controller.pipe("delete", move.x, move.y)

    def bulk_update(self, tuples):
        """
        tuples  -- [ (color1, x1, y1), (color2, x2, y2), ... ]  a list of moves. set color to E to remove stone

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
        """
        Entry point to provide corrections made by the user to stone(s) location(s) on the Goban. See _learn().

        """
        try:
            self.corrections.put_nowait((err_move, exp_move))
        except queue.Full:
            print("Corrections queue full (%s), ignoring %s -> %s" % (correc_size, str(err_move), str(exp_move)))

    def is_empty(self, x: int, y: int) -> bool:
        """
        Return true if the (x, y) goban position is empty (color = E).

        """
        return self.vmanager.controller.is_empty_blocking(y, x)

    def _empties(self) -> (int, int):
        """
        Yields the unoccupied positions of the goban in naive order.
        Note: this implementation allows for the positions to be updated by another thread during yielding.

        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.is_empty_blocking(x, y):
                    yield y, x

    def _empties_spiral(self) -> (int, int):
        """
        Yields the unoccupied positions of the goban along an inward spiral.
        Aims to help detect hand / arm appearance faster by analysing outer border(s) first.

        """
        inset = 0
        while inset <= gsize / 2:
            for x, y in self._empties_border(inset):
                yield x, y
            inset += 1

    def _empties_border(self, inset):
        """
        Yields the unoccupied positions of the goban along an inward spiral.

        inset -- the inner margin defining the start of the inward spiral [0=outer border -> gsize/2=center position].
                 it always ends at the center of the goban.

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
        """
        Return the rectangle of pixels around the provided goban intersection (r, c).
        This method relies on self._posgrid.mtx to get the coordinates of the intersections (so they apply in an
        image of the same size as the canonical frame).

        r -- the intersection row index
        c -- the intersection column index
        cursor -- must be float, has sense in the interval ]0, 2[
                  0 -> the zone is restricted to the (r, c) point.
                  2 -> the zone is delimited by the rectangle (r-1, c-1), (r+1, c+1).
                  1 -> the zone is a rectangle of "intuitive" size, halfway between the '0' and '2' cases.

        return -- x0, y0, x1, y1  the rectangle

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
        """
        A boolean mask shaped in "cvconf.canonical_size" that has a circle around each goban intersection.
        Multiply a frame by this mask to zero-out anything outside the circles.

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
        Return a copy of the current goban state, in the numpy coordinates system.

        """
        return self.vmanager.controller.get_stones()

    def get_foreground(self):
        if hasattr(self, "_fg"):
            current, target = self.total_f_processed, self.bg_init_frames
            if current < target:
                print("Warning : background model still initializing ({} / {})".format(current,  target))
            return self._fg
        else:
            raise ValueError("This StonesFinder doesn't seem to be segmenting background. See self.__init__()")

    def find_intersections(self, img: np.ndarray, canvas: np.ndarray=None) -> np.ndarray:
        """
        Return a matrix indicating which intersections are likely to be empty.

        The search is based on hough lines detection: if good lines are found inside the intersection zone, it is
        very unlikely that a stone would be present.

        @todo additional objective: adjusting the position of the grid (some more work ahead for it to work) if a good
        intersection of lines inside a zone is found

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
        """
        A cached wrapper of self.find_intersections()

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
        """
        The approximation of the radius (in pixels) of a stone in the canonical frame.
        Simple implementation based on self._posgrid.size.

        @rtype float
        """
        return self._posgrid.size / gsize / 2

    def stone_boxarea_bounds(self) -> (float, float):
        """
        @return: minimum and maximum areas of a contour's bounding box, that may be candidate to be a stone.
        These bounds are around the default estimated box area for a stone: (2 * self.stone_radius()) ** 2.

        @rtype: (float, float)
        """
        radius = self.stone_radius()
        min_area = (4 / 3 * radius) ** 2
        max_area = (3 * radius) ** 2
        return min_area, max_area

    def check_against(self, stones: np.ndarray, reference: np.ndarray=None, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """
        Check that the newly found 'stones' array is coherent with the already existing 'reference' array.
        'stones' and 'reference' can be subregions of the Goban, as long as they have the same shape.

        Return -1, 0, 1 if the check is respectively refused, undetermined, or passed.

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
        """
        Check that the provided "stones" 2D array is coherent with lines detection in the image: no line should
        be found in zones where a stone has been detected.

        A match is counted for the zone if it is empty (E) and at least one line has also been detected in that zone.

        stones -- a 2D array that can store the objects, used to record the stones found. It is created if not provided.
        grid -- a 3D (or 2D) array that has negative values where lines have been found.
        rs, re -- optional row start and end, can be used to restrain check to a subregion
        cs, ce -- optional column start and end, can be used to restrain check to a subregion

        Return:
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
        """
        Check that the provided "stones" 2D array doesn't contain "big chunks" that wouldn't make sense in a regular
        game of Go.

        rs, re -- optional row start and end, can be used to restrain check to a subregion
        cs, ce -- optional column start and end, can be used to restrain check to a subregion

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
        """
        Check that newly added stones colors match expectations.
        If multiple new stones : there should be at most one more stone of a color than the other color.

        rs, re -- optional row start and end, can be used to restrain check to a subregion
        cs, ce -- optional column start and end, can be used to restrain check to a subregion

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
        """
        Return all the stones on the first line that have no neighbour in a 2-lines thick square around them.

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
        """
        Draw a circle around each intersection of the goban, as they are currently estimated.

        """
        if self._posgrid is not None:
            for i in range(19):
                for j in range(19):
                    p = self._posgrid.mtx[i][j]
                    cv2.circle(img, (p[1], p[0]), 5, (255, 255, 0))

    def _drawvalues(self, img, values):
        """
        Display one value per goban position. Obviously values will soon overlap if they are longish.

        """
        for row in range(gsize):
            for col in range(gsize):
                x, y = self._posgrid.mtx[row, col]
                imgutil.draw_str(img, str(values[row, col]), x - 10, y + 2)

    def draw_stones(self, stones: np.ndarray, canvas: np.ndarray=None):
        """
        Dev method to see an array of stones in an image. It is a simpler alternative than suggesting them to the goban,
        since there's no game logic involved (easier to update on the flight).

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
        """
        Dev method to see how the intersection analysis is doing.
        There's been some care given to line and column visual differentiation, hope it makes sens :)

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
        factor = 0.5 if i % 2 else 1
        blue = 40 / factor
        green = (230 if j % 2 else 10) * factor
        red = (10 if j % 2 else 200) * factor
        return blue, green, red

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_frequ=2):
        """
        Override to take control of the location of the window of this stonesfinder

        """
        if loc is None:
            from camkifu.config.cvconf import sf_loc
            loc = sf_loc
        super()._show(img, name, latency, thread, loc=loc, max_frequ=max_frequ)

    def display_bg_sampling(self, goban_img):
        """
        Display a "message" image indicating the background sampling is running.

        """
        black = np.zeros((goban_img.shape[0], goban_img.shape[1]), dtype=np.uint8)
        message = "BACKGROUND SAMPLING ({0}/{1})".format(self.total_f_processed, self.bg_init_frames)
        imgutil.draw_str(black, message)
        self._show(black)


def update_grid(lines, box, result_slot):
    """
    Analyse the lines, in the context of the zone:
        - if there's only one valid line, or too many lines, mark the zone as empty (negate result_slot).
        - if there is a decent (1 < x < 5) number of lines, compute the center of mass of their intersections
          that are located inside the zone (including a safety margin). then update result_slot accordingly
          with the new values, negated as well to indicate the zone is probably empty.

    Short version : negative values indicate that the zone (probably) contains no stone. The negative values
    may have been updated as well.

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
        number = 0
        for seg1 in segments:
            for seg2 in segments:
                if seg1 is not seg2:
                    p = seg1.intersection(seg2)
                    if p is not None:
                        p = p[1] + box[0], p[0] + box[1]  # add offset to match global image coordinates
                        if imgutil.within_margin(p, box, margin):
                            x_sum += p[0]
                            y_sum += p[1]
                            number += 1
        if 0 < number:
            result_slot[0] = - x_sum / number
            result_slot[1] = - y_sum / number


class PosGrid(object):
    """
    Store the location of each intersection of the goban, in "numpy coordinates frame" format.

    -- size : the length in pixels of one side of the goban canonical frame (supposed to be a square for now).

    """

    def __init__(self, size):
        self.size = size
        self.mtx = np.zeros((gsize, gsize, 2), dtype=np.int16)  # stores the pixel position of each intersection o the goban
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
        """
        Find the closest intersection from the given (x,y) point of the canonical image (goban image).
        Note : point coordinates are expected in numpy coordinates system.

        Return the closest goban row and column, both in [0, gsize[  ([0, 18[), also in numpy coordinates system.

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
        """
        Update the current grid positions: compute the mean diff vector between the current mtx and the provided grid.
        Then shift all the positions of the current grid by this vector. The diff vector is multiplied by factor before
        being applied.

        """
        assert 0 < rate <= 1  # if 0, why call ?
        diff = grid - self.mtx
        if np.min(diff) < -200:
            raise ValueError("Provided grid seems too far from original, at least for one point.")
        vect = np.sum(diff, axis=(0, 1), dtype=np.float32)
        # then, normalize.
        # in theory, should be the count of points that moved in at least one direction. but that's good enough for now.
        contributors = np.count_nonzero(np.absolute(diff[:, :, 0]) + np.absolute(diff[:, :, 1]))
        vect /= contributors
        self.adjust_vect *= (1.0 - rate)
        self.adjust_vect += vect * rate
        self.adjust_contribs += contributors
        if 20 < self.adjust_contribs:
            print("Grid adjust vector : {}".format(self.adjust_vect))
            self.mtx += self.adjust_vect.astype(np.int16)
            self.adjust_vect[:] = 0
            self.adjust_contribs = 0
