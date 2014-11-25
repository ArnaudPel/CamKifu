from math import pi
from queue import Queue, Full
import cv2
from time import time

from numpy import zeros, uint8, int16, float32, sum as npsum, empty, ogrid
from numpy.core.multiarray import count_nonzero
from numpy.ma import absolute, empty_like

from camkifu.config.cvconf import canonical_size, sf_loc
from camkifu.core.imgutil import draw_circles, draw_str, Segment
from camkifu.core.video import VidProcessor
from golib.config.golib_conf import gsize, E
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
        self.total_f_processed = 0  # total number of frames processed since init. Dev var essentially.

    def _doframe(self, frame):
        transform = None
        if self.vmanager.board_finder is not None:
            transform = self.vmanager.board_finder.mtx
        if transform is not None:
            goban_img = cv2.warpPerspective(frame, transform, (canonical_size, canonical_size))
            self._learn()
            self._find(goban_img)
            self.total_f_processed += 1
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

    def suggest(self, color, x, y, ctype='np'):
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

    def corrected(self, err_move, exp_move):
        """
        Entry point to provide corrections made by the user to stone(s) location(s) on the Goban. See _learn().

        """
        try:
            self.corrections.put_nowait((err_move, exp_move))
        except Full:
            print("Corrections queue full (%s), ignoring %s -> %s" % (correc_size, str(err_move), str(exp_move)))

    def is_empty(self, x, y):
        """
        Return true if the (x, y) goban position is empty (color = E).

        """
        return self.vmanager.controller.rules[y][x] == E

    def _empties(self):
        """
        Yields the unoccupied positions of the goban in naive order.
        Note: this implementation allows for the positions to be updated by another thread during yielding.

        """
        for x in range(gsize):
            for y in range(gsize):
                if self.vmanager.controller.rules[x][y] == E:
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
            # todo extract "do_yield()" method to remove code duplicate ?
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        x = gsize - inset - 1
        for y in range(inset + 1, gsize - inset):
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        y = gsize - inset - 1
        for x in range(gsize - inset - 2, inset - 1, -1):  # reverse just to have a nice spiral. not actually useful
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

        x = inset
        for y in range(gsize - inset - 2, inset, -1):
            if self.vmanager.controller.rules[x][y] == E:
                yield y, x

    def getcolors(self):
        """
        Return a copy of the current goban state.

        """
        return self.vmanager.controller.rules.copystones()

    def _getzone(self, img, r, c, cursor=1.0):
        """
        Returns the (rectangle) pixel zone corresponding to the given goban intersection.

        img -- expected to contain the goban pixels only, in the canonical frame.
        r -- the intersection row index
        c -- the intersection column index
        cursor -- must be float, has sense in the interval ]0, 2[
                  0 -> the zone is restricted to the (r, c) point.
                  2 -> the zone is delimited by the rectangle (r-1, c-1), (r+1, c+1).
                  1 -> the zone is a rectangle of "intuitive" size, halfway between the '0' and '2' cases.

        """
        assert isinstance(cursor, float)
        p = self._posgrid.mtx[r][c]
        pbefore = self._posgrid.mtx[r - 1][c - 1].copy()
        pafter = self._posgrid.mtx[min(r + 1, gsize - 1)][min(c + 1, gsize - 1)].copy()
        if r == 0:
            pbefore[0] = -p[0]
        elif r == gsize - 1:
            pafter[0] = 2 * img.shape[0] - p[0] - 2
        if c == 0:
            pbefore[1] = -p[1]
        elif c == gsize - 1:
            pafter[1] = 2 * img.shape[1] - p[1] - 2

        # determine start and end point of the rectangle
        w = cursor / 2
        sx = max(0, int(w * pbefore[0] + (1 - w) * p[0]))
        sy = max(0, int(w * pbefore[1] + (1 - w) * p[1]))
        ex = min(img.shape[0], int((1 - w) * p[0] + w * pafter[0]))
        ey = min(img.shape[1], int((1 - w) * p[1] + w * pafter[1]))

        # todo remove this copy() and leave it to caller
        return img[sx: ex, sy: ey].copy(), (sx, sy, ex, ey)

    # todo see if still needed
    def getmask(self, shape):
        """
        A boolean mask the size of "frame" that has a circle around each goban intersection.
        Multiply a frame by this mask to zero-out anything outside the circles.

        """
        if self.mask_cache is None:
            # todo : observation shows that stones of the front line are seen too high (due to cam angle most likely)
            # investigate more and see to adapt the repartition of the mask ? Some sort of vertical gradient of size or
            # location. The former will imply the introduction of a structure to store all zones areas, at least one
            #  per line.
            print("initializing mask")
            self.mask_cache = empty(shape)
            mask = empty(shape[0:2], dtype=uint8)
            for row in range(gsize):
                for col in range(gsize):
                    zone, (sx, sy, ex, ey) = self._getzone(mask, row, col)  # todo expose proportions ?
                    a = zone.shape[0] / 2
                    b = zone.shape[1] / 2
                    r = min(a, b)
                    y, x = ogrid[-a:zone.shape[0] - a, -b: zone.shape[1] - b]
                    zmask = x * x + y * y <= r * r
                    mask[sx: ex, sy: ey] = zmask

            # duplicate mask to match image depth
            if len(shape) == 3:
                for i in range(self.mask_cache.shape[2]):
                    self.mask_cache[:, :, i] = mask
            else:
                self.mask_cache[:] = mask

            # store the area of one zone for normalizing purposes
            zone, _ = self._getzone(mask, 0, 0)
            self.zone_area = npsum(zone)
            print("area={0}".format(self.zone_area))

        return self.mask_cache

    def _drawgrid(self, img):
        """
        Draw a circle around each intersection of the goban, as they are currently estimated.

        """
        if self._posgrid is not None:
            centers = []
            for i in range(19):
                for j in range(19):
                    centers.append(self._posgrid.mtx[i][j])
            draw_circles(img, centers)

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

    def search_intersections(self, img):
        """
        Return a matrix indicating which intersections are likely to be empty.

        The search is based on hough lines detection: if good lines are found inside the intersection zone, it is
        very unlikely that a stone would be present.

        @todo additional objective: adjusting the position of the grid (some more work ahead for it to work) if a good
        intersection of lines inside a zone is found

        """
        canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(canny, 25, 75)
        # noinspection PyNoneFunctionAssignment
        grid = self._posgrid.mtx.copy()
        for r in range(gsize):
            for c in range(gsize):
                zone, points = self._getzone(canny, r, c)
                min_side = min(zone.shape[0], zone.shape[1])
                thresh = int(min_side * 3 / 4)
                min_len = int(min_side * 2 / 3)
                lines = cv2.HoughLinesP(zone, 1, pi / 180, threshold=thresh, maxLineGap=0, minLineLength=min_len)
                if lines is not None and 0 < len(lines):
                    self.update_grid(lines, points, zone, grid[r][c])
        return grid

    @staticmethod
    def update_grid(lines, points, zone, result_slot):
        """
        Analyse the lines, in the context of the zone:
            - if there's only one valid line, or too many lines, mark the zone as empty (negate result_slot).
            - if there is a decent (1 < x < 5) number of lines, compute the center of mass of their intersections
              that are located inside the zone (including a safety margin). then update result_slot accordingly
              with the new values, negated as well to indicate the zone is probably empty.

        Short version : negative values indicate that the zone (probably) contains no stone. The negative values
        may have been updated as well.

        """
        # step one: only retain lines that are either vertical or horizontal enough
        segments = []
        for line in lines:
            seg = Segment(line[0], zone)
            if seg.slope < 0.1:  # horizontal / vertical check
                seg.set_offset(points[1], points[0])  # swap "points" to opencv coord
                segments.append(seg)
        # step two, analyse filtered lines
        if len(segments):
            # if at least one line present, indicates that the intersection is occupied
            # todo check that the line is inside the same borders as below to ignore tangents around stones
            result_slot[:] *= -1
            # pass
        # then if there's a decent amount of lines, try to refine intersection location
        if 1 < len(segments) < 5:
            x_sum = 0
            y_sum = 0
            number = 0
            margin = min(zone.shape[0], zone.shape[1]) / 7
            for seg1 in segments:
                for seg2 in segments:
                    if seg1 is not seg2:
                        i = seg1.intersection(seg2)
                        if i is not None:
                            i = i[0] + points[1], i[1] + points[0]  # add offset to match global image coordinates
                            if points[1] + margin < i[0] < points[3] - margin \
                                    and points[0] + margin < i[1] < points[2] - margin:
                                x_sum += i[0]
                                y_sum += i[1]
                                number += 1
            if 0 < number:
                result_slot[0] = - y_sum / number
                result_slot[1] = - x_sum / number

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
        self._show(img)

    @staticmethod
    def get_display_color(i, j):
        factor = 0.5 if i % 2 else 1
        blue = 40 / factor
        green = (230 if j % 2 else 10) * factor
        red = (10 if j % 2 else 200) * factor
        return blue, green, red


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

    def get_intersection(self, point):
        """
        Return the closest intersection from the given (x,y) point of the canonical frame (goban frame).
        Note : point coordinates are given in image coordinates frame (opencv, numpy), and this method will
        return the converted numbers as (y, x), to be ready for the goban.

        """
        # a simpler version would use the fact that self.learn() shifts the whole grid to store and apply the offset
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
        return target[1], target[0]  # invert to match goban coordinates frame

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
