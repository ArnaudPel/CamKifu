from math import pi, cos, sin
from queue import Queue, Full
from time import time

import cv2
from numpy import zeros, uint8, int16, float32, sum as npsum, empty, ogrid, ndarray
from numpy.core.multiarray import count_nonzero
from numpy.ma import absolute

from camkifu.config.cvconf import canonical_size, sf_loc
from camkifu.core.imgutil import draw_circles, draw_str, Segment, within_margin
from camkifu.core.video import VidProcessor
from golib.config.golib_conf import gsize, E, B, W
from golib.model.move import Move


__author__ = 'Arnaud Peloquin'


correc_size = 10


class StonesFinder(VidProcessor):
    """
    Abstract class providing a base structure for stones-finding processes.
    It relies on the providing of a transform matrix to extract only the goban pixels from the global frame.

    """

    def __init__(self, vmanager):
        super(StonesFinder, self).__init__(vmanager)
        self._posgrid = PosGrid(canonical_size)
        self.mask_cache = None
        self.zone_area = None  # the area of a zone # (non-zero pixels of the mask)
        self.corrections = Queue(correc_size)

    def _doframe(self, frame):
        transform = None
        if self.vmanager.board_finder is not None:
            transform = self.vmanager.board_finder.mtx
        if transform is not None:
            goban_img = cv2.warpPerspective(frame, transform, (canonical_size, canonical_size))
            self._learn()
            self._find(goban_img)
        else:
            if 1 < time() - self.last_shown:
                black = zeros((canonical_size, canonical_size), dtype=uint8)
                draw_str(black, "NO BOARD LOCATION AVAILABLE", int(black.shape[0] / 2 - 110), int(black.shape[1] / 2))
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

    def _learn(self):
        """
        Process corrections queue, and perform algorithm adjustments if necessary.

        This choice to "force" implementation using an abstract method is based on the fact that stones
        deleted by the user MUST be acknowledged and dealt with, in order not to be re-suggested straight away.
        Added stones are not so important, because their presence is automatically reflected in self.empties().

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def _window_name(self):
        return "camkifu.stone.stonesfinder.StonesFinder"

    def suggest(self, color, x: int, y: int, ctype: str='np'):
        """
        Suggest the add of a new stone to the goban.
        -- color : in (E, B, W)
        -- x, y : the coordinates
        -- ctype : the type of coordinates frame (see Move._interpret()), defaults to 'np' (numpy)

        """
        move = Move(ctype, ctuple=(color, x, y))
        print(move)
        self.vmanager.controller.pipe("append", [move])

    def remove(self, x, y, ctype='np'):
        """
        Although allowing automated removal of stones doesn't seem to be a very safe idea given the current
        robustness of stones finders, here's an implementation.

        x, y -- the location in numpy coordinates frame.
        ctype -- the type of coordinates frame (see Move._interpret()), defaults to 'np' (numpy)

        """
        assert not self.is_empty(x, y), "Can't remove stone from empty intersection."
        move = Move(ctype, ("", x, y))
        print("delete {}".format(move))
        self.vmanager.controller.pipe("delete", (move.x, move.y))

    def bulk_update(self, tuples, ctype='np'):
        """
        tuples  -- [ (color1, x1, y1), (color2, x2, y2), ... ]  a list of moves. set color to E to remove stone

        """
        moves = []
        for color, x, y in tuples:
            if color is E and not self.is_empty(x, y):
                    moves.append(Move(ctype, (color, x, y)))
            elif color in (B, W):
                if not self.is_empty(x, y):
                    existing_mv = self.vmanager.controller.locate(y, x)
                    if color is not existing_mv.color:  # if existing_mv is None, better to crash now, so no None check
                        # delete current stone to be able to put the other color
                        moves.append(Move(ctype, (E, x, y)))
                    else:
                        continue  # already up to date, go to next iteration
                moves.append(Move(ctype, (color, x, y)))
        if len(moves):
            self.vmanager.controller.pipe("bulk", [moves])

    def corrected(self, err_move, exp_move) -> None:
        """
        Entry point to provide corrections made by the user to stone(s) location(s) on the Goban. See _learn().

        """
        try:
            self.corrections.put_nowait((err_move, exp_move))
        except Full:
            print("Corrections queue full (%s), ignoring %s -> %s" % (correc_size, str(err_move), str(exp_move)))

    def is_empty(self, x, y) -> bool:
        """
        Return true if the (x, y) goban position is empty (color = E).

        """
        return self.vmanager.controller.is_empty_blocking(y, x)

    def _empties(self):
        """
        Yields the unoccupied positions of the goban in naive order.
        Note: this implementation allows for the positions to be updated by another thread during yielding.

        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.is_empty_blocking(x, y):
                    yield y, x

    def _empties_spiral(self):
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

    def _getrect(self, r: int, c: int, cursor: float=1.0) -> (int, int, int, int):
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

    def getmask(self, shape: tuple) -> ndarray:
        """
        A boolean mask the size of "frame" that has a circle around each goban intersection.
        Multiply a frame by this mask to zero-out anything outside the circles.

        """
        # todo remove parameter shape to give more sense to caching ? most likely this mask can only apply to the
        # canonical frame.
        if self.mask_cache is None or self.mask_cache.shape != shape:
            # todo : observation shows that stones of the front line are seen too high (due to cam angle most likely)
            # investigate more and see to adapt the repartition of the mask ? Some sort of vertical gradient of size or
            # location. The former will imply the introduction of a structure to store all zones areas, at least one
            #  per line.
            print("initializing mask")
            self.mask_cache = empty(shape)
            mask = empty(shape[0:2], dtype=uint8)
            for row in range(gsize):
                for col in range(gsize):
                    x0, y0, x1, y1 = self._getrect(row, col)  # todo expose proportions ?
                    zone = mask[x0:x1, y0:y1]
                    a = zone.shape[0] / 2
                    b = zone.shape[1] / 2
                    r = min(a, b)
                    y, x = ogrid[-a:zone.shape[0] - a, -b: zone.shape[1] - b]
                    zmask = x * x + y * y <= r * r
                    mask[x0:x1, y0:y1] = zmask

            # duplicate mask to match image depth
            if len(shape) == 3:
                for i in range(self.mask_cache.shape[2]):
                    self.mask_cache[:, :, i] = mask
            else:
                self.mask_cache[:] = mask

            # store the area of one zone for normalizing purposes
            x0, y0, x1, y1 = self._getrect(0, 0)
            self.zone_area = npsum(mask[x0:x1, y0:y1])
            print("area={0}".format(self.zone_area))
        return self.mask_cache

    def _drawgrid(self, img: ndarray):
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
                draw_str(img, str(values[row, col]), x - 10, y + 2)

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_frequ=2):
        """
        Override to take control of the location of the window of this stonesfinder

        """
        location = sf_loc if loc is None else loc
        super()._show(img, name, latency, thread, loc=location, max_frequ=max_frequ)

    def search_intersections(self, img: ndarray) -> ndarray:
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
                x0, y0, x1, y1 = self._getrect(r, c)
                zone = canny[x0:x1, y0:y1]
                min_side = min(zone.shape[0], zone.shape[1])
                thresh = int(min_side * 3 / 4)
                min_len = int(min_side * 2 / 3)
                lines = cv2.HoughLinesP(zone, 1, pi / 180, threshold=thresh, maxLineGap=0, minLineLength=min_len)
                if lines is not None and 0 < len(lines):
                    # todo display lines again to see why adjusting grid seems to lower the number of "crosses" found
                    update_grid(lines, (x0, y0, x1, y1), grid[r][c])
        return grid

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
        seg = Segment(line[0])
        if 0.995 < abs(cos(seg.theta)):  # horizontal check + margin respect
            p = (box[0] + box[2]) / 2, (seg.p1()[0] + seg.p2()[0]) / 2 + box[1]
            if not within_margin(p, box, margin):
                continue
        elif 0.995 < abs(sin(seg.theta)):  # vertical check + margin respect
            p = (seg.p1()[1] + seg.p2()[1]) / 2 + box[0], (box[3] + box[1]) / 2  # numpy coordinates
            if not within_margin(p, box, margin):
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
                        if within_margin(p, box, margin):
                            x_sum += p[0]
                            y_sum += p[1]
                            number += 1
        if 0 < number:
            result_slot[0] = - x_sum / number
            result_slot[1] = - y_sum / number


def evalz(zone, chan):
    """ Return an integer evaluation for the zone. """
    return int(npsum(zone[:, :, chan]))


def compare(reference, current):
    """
    Return a distance between the two colors. The value is positive if current is
    brighter than the reference, and negative otherwise.

    reference -- a vector, usually of shape (3, 1)
    current -- a vector of same shape as reference.

    """
    sign = 1 if npsum(reference) <= npsum(current) else -1
    return sign * int(npsum(absolute(current - reference)))


class PosGrid(object):
    """
    Store the location of each intersection of the goban, in "numpy coordinates frame" format.

    -- size : the length in pixels of one side of the goban canonical frame (supposed to be a square for now).

    """

    def __init__(self, size):
        self.size = size
        self.mtx = zeros((gsize, gsize, 2), dtype=int16)  # stores the pixel position of each intersection o the goban
        self.adjust_vect = zeros(2, dtype=float32)
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
                        dist = sum(absolute(self.mtx[x][y] - point))
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
        vect = npsum(diff, axis=(0, 1), dtype=float32, keepdims=True)
        # then, normalize.
        # in theory, should be the count of points that moved in at least one direction. but that's good enough for now.
        contributors = count_nonzero(absolute(diff[:, :, 0]) + absolute(diff[:, :, 1]))
        vect /= contributors
        self.adjust_vect *= (1.0 - rate)
        self.adjust_vect += vect * rate
        self.adjust_contribs += contributors
        if 20 < self.adjust_contribs:
            print("Adjust vector : {}".format(self.adjust_vect))
            self.mtx += self.adjust_vect.astype(int16)
            self.adjust_vect[:] = 0
            self.adjust_contribs = 0
