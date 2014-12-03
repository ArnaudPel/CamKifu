from bisect import insort
import cv2
import random
from numpy import vstack, float32
from camkifu.core.imgutil import Segment

__author__ = 'Kohistan'

"""
Legacy bits that may well go away sometime. Things here are not supposed to be used.

"""


def runmerge(grid):
    """
    Legacy function (sub-functions included), used here class mostly because they were at close hand
    at dev time. The 200ish lines of code taking the form of (_merge, _least_squares, _get_neighbours,
    _error, _get_seg) would most likely benefit from deletion and global rethink of the automated board_finder.

    """
    merged = SegGrid([], [], grid.img)
    discarded = SegGrid([], [], grid.img)

    # merge locally first to minimize askew lines
    for i in range(2):  # i=0 is horizontal, i=1 is vertical
        low = SegGrid([], [], grid.img)
        high = SegGrid([], [], grid.img)
        mid = grid.img.shape[1 - i] / 2  # numpy (i,j) is equivalent to opencv (y,x)
        for seg in (grid.vsegs if i else grid.hsegs):
            if (seg.coords[0 + i] + seg.coords[2 + i]) / 2 < mid:
                low.insort(seg)
            else:
                high.insort(seg)
        mgd, disc = _merge(low)
        merged += mgd
        discarded += disc
        mgd, disc = _merge(high)
        merged += mgd
        discarded += disc

    # run global merges with increasing tolerance for error
    merged += discarded
    for precision in (1, 4, 16, 32):
        merged, discarded = _merge(merged, precision=precision)
        merged += discarded
    return merged


def _merge(grid, precision=1):
    """
    Merge segments that seem to appear to the same line. A merge between two segments is
    a least-square segment of the 4 points being merged, as per cv2.fitLine().

    Returns:
        merged_grid, discarded_grid
    Args:
        precision -- the max "dispersion" allowed around the regressed line to accept a merge.

    """

    merged = SegGrid([], [], grid.img)
    discarded = SegGrid([], [], grid.img)
    random.seed(42)
    correction = 0

    for i in range(2):
        if i:
            segments = list(grid.vsegs)  # second pass, vertical lines
        else:
            segments = list(grid.hsegs)  # first pass, horizontal lines

        while 1 < len(segments):
            i = random.randint(0, len(segments) - 1)
            seg = segments.pop(i)
            valuations = []

            # valuate against the 'n' closest neighbours
            for neighb in _get_neighbours(segments, i, seg.intercept):
                _least_squares(seg, neighb, valuations)
                if 0 < len(valuations) and valuations[0][0] < 0.1:  # error small enough already, don't loop
                    break

            if 0 < len(valuations):
                bestmatch = valuations[0]
                if bestmatch[0] < precision:  # if error acceptable
                    segments.remove(bestmatch[1])
                    segmt = Segment(_get_seg(bestmatch[2]))
                    merged.insort(segmt)
                else:
                    discarded.insort(seg)
            else:
                discarded.insort(seg)

        # last segment has not been merged, but is not necessarily bad either
        if 0 < len(segments):
            merged.insort(segments[0])
            correction -= 1  # -1 because last segment of this pass has not been merged
    assert len(grid) == 2 * len(merged) + len(discarded) + correction
    return merged, discarded


def _least_squares(seg, neighb, valuations):
    """
    Merge "seg" and "neighb", and insert the resulting segment into "valuations",
    ordering on the regression error.

    """
    p1 = seg.coords[0:2]
    p2 = seg.coords[2:4]
    p3 = neighb.coords[0:2]
    p4 = neighb.coords[2:4]
    ndarray = vstack([p1, p2, p3, p4])
    points = float32(ndarray)
    regression = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    error, projections = _error(points, regression)
    insort(valuations, (error, neighb, projections))


def _get_neighbours(segments, start, intercept):
    """
    Generator that returns segments whose intercept is increasingly more distant from
        the "intercept" argument. The list is supposed to be sorted, and "start" is supposed
        to be the position of "intercept" in the list.

    """
    l = start - 1
    r = start
    stop = len(segments)
    while 0 <= l or r < stop:

        # get neighbours in both directions
        if 0 <= l:
            left = segments[l]
        else:
            left = None
        if r < stop:
            right = segments[r]
        else:
            right = None

        # select best direction
        if right is None:
            yield left
            l -= 1
            continue
        if left is None:
            yield right
            r += 1
            continue
        ldiff = intercept - left.intercept
        rdiff = right.intercept - intercept

        if 10 < min(ldiff, rdiff):  # improvement: tune pixel dist according to img.shape ?
            return  # don't explore neighbours whose intercept is more than 'n' pixels away

        if ldiff < rdiff:
            yield left
            l -= 1
        else:
            yield right
            r += 1


def _error(points, regr):
    """
    >>> points = [(1.0, 0.0)]
    >>> regression = [[0.7071067811865475], [0.7071067811865475], [0.0], [0.0]]
    >>> print _error(points, regression)
    (0.5, [[0.5, 0.5]])

    >>> points = [(-4.0, 5.0)]
    >>> regression = [[1.0], [5.0/3], [0.0], [-5.0]]
    >>> print _error(points, regression)
    (73.529411764705884, [[3.3529411764705888, 0.58823529411764763]])

    """
    vx = regr[0][0]
    vy = regr[1][0]
    x0 = regr[2][0]
    y0 = regr[3][0]
    # column vectors for matrix calculation
    vect = vstack([vx, vy])
    p0 = vstack([x0, y0])

    projector = vect.dot(vect.T) / vect.T.dot(vect)  # projection matrix

    error = 0
    projections = []
    for point in points:
        actual = vstack([point[0], point[1]])  # make sure we have column vector here as well
        projection = projector.dot(actual - p0) + p0
        errvect = actual - projection
        err = errvect.T.dot(errvect)
        error += err
        projections.append([c[0] for c in projection])
    return error[0][0], projections


def _get_seg(points):
    """
    Return the two most distant points from each other of the given list.
    note: because the expected number of points is 4 in this context, the most
    basic comparison has been used, in o(n2)

    points -- any collection of (x,y) couples having [] defined should work.

    >>> points = [(2, 0), (0, 1), (-1, 0), (0, -1)]
    >>> print _get_seg(points)
    (-1, 0, 2, 0)

    """
    if len(points) < 3:
        return points

    seg = None
    dist = 0
    for i in range(1, len(points)):
        p0 = points[i]
        for j in range(i):
            p1 = points[j]
            curdist = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2
            if dist < curdist:
                seg = (p0[0], p0[1], p1[0], p1[1])
                dist = curdist
    return seg


class SegGrid:
    """
    A structure that store line segments in two different categories: horizontal and vertical.
    This implementation is not good as it should instead make two categories based on their
    reciprocal orthogonality.

    These two groups can represent the horizontal and vertical lines of a goban.

    """
    def __init__(self, hsegs, vsegs, img):
        assert isinstance(hsegs, list), "hsegs must be of type list."
        assert isinstance(vsegs, list), "vsegs must be of type list."
        self.hsegs = hsegs
        self.vsegs = vsegs
        self.img = img

    def __add__(self, other):
        assert isinstance(other, SegGrid), "can't add: other should be a grid."
        assert self.img.shape == other.img.shape, "images should have same shape when adding grids."
        hsegs = [seg for seg in self.hsegs + other.hsegs]
        vsegs = [seg for seg in self.vsegs + other.vsegs]
        hsegs.sort()
        vsegs.sort()
        return SegGrid(hsegs, vsegs, self.img)

    def __iter__(self):
        return SegGridIter(self)

    def __len__(self):
        return len(self.hsegs) + len(self.vsegs)

    def __str__(self):
        rep = "Grid(hsegs:" + str(len(self.hsegs))
        rep += ", vsegs:" + str(len(self.vsegs)) + ")"
        return rep

    def enumerate(self):
        return self.hsegs + self.vsegs

    def insort(self, segment):
        insort(self.hsegs, segment) if segment.horiz else insort(self.vsegs, segment)


class SegGridIter(object):
    """
    Iterator used in SegGrid.__iter__()

    """
    def __init__(self, grid):
        self.grid = grid
        self.idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        l1 = len(self.grid.hsegs)
        if self.idx < l1:
            return self.grid.hsegs[self.idx]
        elif self.idx - l1 < len(self.grid.vsegs):
            return self.grid.vsegs[self.idx - l1]
        else:
            assert self.idx == len(self.grid), "Should describe entire grid once and only once."
            raise StopIteration


def split_sq(img, nbsplits=5, offset=False):
    """
    Split the image in nbsplits * nbsplits squares and yield each Chunk.

    """
    for hchunk in split_h(img, nbsplits, offset=offset):
        for vchunk in split_v(hchunk.mat, nbsplits, offset=offset):
            yield Chunk(hchunk.x + vchunk.x, hchunk.y + vchunk.y, vchunk.mat, img)


def split_v(img, nbsplits=5, offset=False):
    """
    Split the image in "nbsplits" vertical strips, and yields the Chunks.
    nbsplits -- the number of strips required.

    """
    for i in range(nbsplits):
        strip_size = img.shape[1] / nbsplits
        offs = strip_size / 2 if offset else 0
        x0 = i * strip_size + offs

        if (i + 1) < nbsplits:
            x1 = (i + 1) * strip_size + offs
        else:
            if offset:
                return
            else:
                x1 = img.shape[1]

        y0 = 0
        y1 = img.shape[0]
        yield Chunk(x0, y0, img[y0:y1, x0:x1].copy(), img)


def split_h(img, nbsplits=5, offset=False):
    """
    Split the image in "nbsplits" horizontal strips, and yields the Chunks.
    nbsplits -- the number of strips required.

    """

    for i in range(nbsplits):
        x0 = 0
        x1 = img.shape[1]
        strip_size = img.shape[0] / nbsplits
        offs = strip_size / 2 if offset else 0
        y0 = i * strip_size + offs

        if (i + 1) < nbsplits:
            y1 = (i + 1) * strip_size + offs
        else:
            if offset:
                return  # skipp last block when offsetting
            else:
                y1 = img.shape[0]

        yield Chunk(x0, y0, img[y0:y1, x0:x1].copy(), img)


class Chunk:
    """
    A chunk (subpart) of an image.

    x -- the x origin of the Chunk inside the source image.
    y -- the y origin of the Chunk inside the source image.
    mat -- the chunk itself, a matrix.
    src -- the source image.
    """

    def __init__(self, x0, y0, mat, source):
        self.x = x0
        self.y = y0
        self.mat = mat
        self.src = source

    def __setitem__(self, key, value):
        pass

    def paint(self, dest):
        """
        Copy values from this Chunk into the global (dest) image, at the right location.

        """
        dest[self.x:self.mat.shape[0] + self.x, self.y:self.mat.shape[1] + self.y] = self.mat