import bisect
import math
import sys
from os.path import isfile

import cv2
import numpy as np

from golib.config import golib_conf


def connect_clusters(groups, dist):
    """ Do one connectivity-based clustering pass.

    Merge groups from which at least one couple of points is closer than 'dist' (the groups are separated by less
    than dist).

    This has been implemented quickly and is most likely not the best way to proceed.

    Args:
        groups: list
            The groups to merge.
        dist: number
            The maximum distance below which two groups should be merged.
    """
    todel = []
    for g0 in groups:
        merge = None
        for p0 in g0:
            for g1 in groups:
                if g0 is not g1 and g1 not in todel:
                    for p1 in g1:
                        if (p0[0] - p1[0]) ** 2 + (p0[0] - p1[0]) ** 2 < dist:
                            merge = g1
                            break
                if merge: break
            if merge: break
        if merge:
            merge.extend(g0)
            todel.append(g0)
    for gdel in todel:
        groups.remove(gdel)


def draw_lines(img, segments, color=255):
    """ Draw each segment on the image.

    Args:
        img: ndarray
            The image on which to draw.
        segments: iterable
            Each segment can either be 4 integers (x0, y0, x1, y1), or two couples of integers ((x0, y0), (x1, y1)).
        color: int or tuple of int
            Can be an int, or a tuple of ints which length matches the depth of 'img'.
    """
    thickness = 1 + 2 * _factor(img)
    for seg in segments:
        if len(seg) == 4:
            p1 = (seg[0], seg[1])
            p2 = (seg[2], seg[3])
        elif len(seg) == 2:
            p1 = seg[0]
            p2 = seg[1]
        else:
            raise ValueError("Unrecognized segment format: {}".format(seg))
        cv2.line(img, p1, p2, color=color, thickness=thickness)


def draw_str(dst, s, x=None, y=None, color=(255, 255, 255)):
    """ Print a white string with a black shadow on the image. (Thank you opencv python samples)

    Args:
        dst: ndarray
            The image on which to draw.
        x: int
            Horizontal offset from left, in pixels. Defaults to a value that tries to center the str.
        y: int
            Vertical offset from top, in pixels. Defaults to a value that tries to center the str.
        s: str
            The string to draw.
        color: tuple
            The BGR color with which the text should be displayed.
    """
    if x is None:
        # try to center horizontally depending on the string length. quite approximate since the font isn't monospaced
        x = int(dst.shape[1] / 2 - len(s) * 3.5)
    if y is None:
        y = int(dst.shape[0] / 2 - 3)
    # the shadow
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # the white text
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, lineType=cv2.LINE_AA)


windows = set()  # dev workaround to center windows the first time they are displayed only


def show(img, name="Camkifu", loc=None):
    """ Show the specified image in its own window.

    Args:
        img: ndarray
            The image to show.
        name: str
            The name of the window.
        loc: (int, int)
            The location of the window to set (as per cv2.moveWindow). Default to a value that tries to center.
    """
    if name not in windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if loc is not None:
            cv2.moveWindow(name, *loc)
        else:
            center = (golib_conf.screenw / 2, golib_conf.screenh / 2)
            cv2.moveWindow(name, max(0, int(center[0] - img.shape[0] / 2)), int(img.shape[1] / 2))
        windows.add(name)
    cv2.imshow(name, img)


def destroy_win(name):
    """ Destroy the window having the provided name.

    Args:
        name: str
            The name of the window to destroy. Raise KeyError if this name has not been shown by the
            show() method implemented in this file.
    """
    windows.remove(name)
    cv2.destroyWindow(name)


def _factor(img):
    """ Find how many times the image should be "pyrDown" to fit inside the screen.

    Returns f: int
        The number of times the image should be pyrdowned.
    """
    f = 0
    imwidth = img.shape[1]
    imheight = img.shape[0]
    while golib_conf.screenw < imwidth or golib_conf.screenh < imheight:
        f += 1
        imwidth /= 2
        imheight /= 2
    return f


def saturate(img):
    """ Multiply Saturation and Value by 1.5 (in HSV space).

    Args:
        img: ndarray
            Supposed to be in the BGR space.
    Returns saturated: ndarray
            The saturated image, in BGR space.
    """
    maxsv = np.ones_like(img)
    maxsv[:, :, 1:3] *= 255
    saturated = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturated[:, :, 1:3] = np.minimum(maxsv[:, :, 1:3], saturated[:, :, 1:3] * 1.5)
    return cv2.cvtColor(saturated, cv2.COLOR_HSV2BGR)


def rgb_histo(img):
    """ Display an RGB histogram for the image.
    Code from: http://opencvpython.blogspot.fr/2012/04/drawing-histogram-in-opencv-python.html

    Args:
        img: ndarray
            The image to sample.
    Returns h: ndarray
        An image representing the histogram (ready to be displayed).
    """
    h = np.zeros((300, 256, 3))
    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    return np.flipud(h)


def segment_from_hough(hough_line, img_shape):
    """ Create a Segment based on an (infinite) line as represented by cv2.HoughLines().

    Args:
        hough_line: ndarray
            The line as returned by cv2.HoughLines(). Only one line not the whole list.
        img_shape: (int, int,...)
            As per img.shape. Gives an order of magnitude for the segment length, since the line from cv2.HoughLines()
            is a mathematical representation that has no beginning or end.
    """
    rho, theta = hough_line[0]
    a, b = math.cos(theta), math.sin(theta)
    x0, y0 = a * rho, b * rho
    extent = max(img_shape[0], img_shape[1])
    pt1 = int(x0 + extent * (-b)), int(y0 + extent * a)
    pt2 = int(x0 - extent * (-b)), int(y0 - extent * a)
    segment = Segment((pt1[0], pt1[1], pt2[0], pt2[1]))
    return segment


def cyclic_permute(points):
    """ Permute the list, until its first element becomes the closest point to the upper-left corner of the image.
    The overall order of the sequence is not modified.

    Args:
        points: iterable
            A sequence of points (x, y).

    >>> cyclic_permute([(5, 367), (638, 364), (126, 96), (514, 92)])
    [(126, 96), (514, 92), (5, 367), (638, 364)]
    >>> cyclic_permute([(638, 364), (126, 96), (514, 92), (5, 367)])
    [(126, 96), (514, 92), (5, 367), (638, 364)]
    >>> cyclic_permute([(126, 96), (638, 364), (514, 92), (5, 367)])
    [(126, 96), (638, 364), (514, 92), (5, 367)]

    """
    permuted = []
    idx = 0
    mind = sys.maxsize
    for i in range(len(points)):
        p = points[i]
        dist = p[0] ** 2 + p[1] ** 2
        if dist < mind:
            mind = dist
            idx = i
    for i in range(idx, idx + len(points)):
        p = points[i % len(points)]
        permuted.append((p[0], p[1]))
    return permuted


def get_ordered_hull(points):
    """ Return the convex hull of the given points.
    Additional conditions are ensured:
    - the points are ordered clockwise
    - the first point is the closest to the origin (upper left corner)

    Args:
        points: list
            The points from which to extract a convex hull.
    Returns hull: ndarray
        The convex hull.

    >>> get_ordered_hull([(5, 367), (638, 364), (126, 96), (514, 92)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]
    >>> get_ordered_hull([(5, 367), (126, 96), (638, 364), (514, 92)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]
    >>> get_ordered_hull([(5, 367), (126, 96), (514, 92), (638, 364)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]

    """
    cvhull = cv2.convexHull(np.vstack(points))
    return cyclic_permute([x[0] for x in cvhull])


def _sort_contours(contours, wclass, area_bounds=None):
    """ See sort_contours_box or sort_contours_circle.
    """
    sortedconts = []
    for i, cont in enumerate(contours):
        wrapped = wclass(cont, i)
        if area_bounds is None or area_bounds[0] < wrapped.area < area_bounds[1]:
            bisect.insort(sortedconts, wrapped)
    return sortedconts


def sort_contours_box(contours, area_bounds=None):
    """ Sort contours by increasing bounding-box area, after having filtered out those outside the provided bounds.

    Args:
        contours: iterable
            A container of contours as returned by cv2.findContours()
        area_bounds: (int, int)
            The minimum and maximum area to allow a contour in the sorted list. Outsiders are silently ignored.

    Returns sorted: list
        A sorted list of BoundingBox objects. The 'position' attribute of each BB object corresponds to the contour's
        position in the provided iterable.
    """
    return _sort_contours(contours, BoundingBox, area_bounds=area_bounds)


def sort_contours_circle(contours, area_bounds=None):
    """ Sort contours by increasing enclosing-circle area, after having filtered out those outside the provided bounds.

    Args:
        contours: iterable
            A container of contours as returned by cv2.findContours()
        area_bounds: (int, int)
            The minimum and maximum area to allow a contour in the sorted list. Outsiders are silently ignored.

    Returns sorted: list
        A sorted list of EnclosingCircle objects. The 'position' attribute of each EC object corresponds to the
        contour's position in the provided iterable.
    """
    return _sort_contours(contours, EnclosingCircle, area_bounds=area_bounds)


def norm(p1, p2):
    """ Return the euclidean norm of the vector defined by points p1 and p2.

    >>> "{:.6f}".format(norm((1, 1), (3, 7)))
    '6.324555'

    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def within_margin(p, box, margin):
    """ Check that the point "p" is located inside the provided 'box', with respect of the provided margin.
    Args:
        p: (int, int)
            (x, y)
        box: (int, int, int, int)
            A rectangle described by two points. Format: (x0, y0, x1, y1)
        margin: number
            The extra safety margin to respect before confirming the point is in the box (i.e. shrink the box).

    Returns within: bool
        True if the point is inside the box.
    """
    return box[0] + margin < p[0] < box[2] - margin and box[1] + margin < p[1] < box[3] - margin


def around(x, y, margin, xmin=None, xmax=None, ymin=None, ymax=None):
    """ Yield the positions around (x, y), forming a square of side (margin * 2 + 1).

    Args:
        x, y: int, int
            The center around which positions should be enumerated. Is excluded from the enumeration.
        margin:
            Defines the side of the square that will be enumerated. Its side is (margin * 2 + 1).
        xmin, xmax, ymin, ymax: int, int, int, int
            Additional limits to the row(s) or column(s) enumerated (making the enumeration a rectangle).

    Yield: Each position as described above.
    """
    for i in range(-margin, margin + 1):
        if (xmin is None or xmin <= x + i) and (xmax is None or x + i < xmax):
            for j in range(-margin, margin + 1):
                if i == j == 0:
                    continue
                if (ymin is None or ymin <= y + j) and (ymax is None or y + j < ymax):
                    yield x + i, y + j


def draw_contours_multicolor(img, contours):
    """ Draw each contour with a color that is different from its neighbours.

    I haven't put much thought into the mathematical guarantee that neighbours won't have the same color. The aim is
    just to help visually differentiate contours that would have appeared as a single contour when drawn with a unique
    color.

    Args:
        img: ndarray
            The image onto which draw.
        contours: list
            The contours as returned by cv2.findContours().
    """
    for i, contour in enumerate(contours):
        b = int(255 * (7 - i % 7) / 7)
        g = int(255 * (i % 5 + 1) / 5)
        r = int(255 * (3 - i % 3) / 3)
        cv2.drawContours(img, contours, i, (b, g, r))


def is_img(argument):
    try:
        return isfile(argument) and (argument.lower().endswith('.png') or argument.lower().endswith('.jpg'))
    except:
        return False


class BoundingBox:
    """ Associate a contour with its bounding rectangle (rotated to fit).

    Attributes:
        contour: ndarray
            The contour as computed by openCV.
        pos: int
            Arbitrary index that can be set to remember the position of this contour in a structure.
        box: tuple
            The bounding rectangle of the contour, as computed by openCV.
        area: int
            The area of the bounding rectangle.
    """

    def __init__(self, contour, pos):
        self.contour = contour
        self.pos = pos
        # self.box = cv2.boundingRect(contour)
        self.box = cv2.minAreaRect(contour)
        self.area = self.box[1][0]*self.box[1][1]

    def __lt__(self, other):
        return self.area < other.area

    def __repr__(self):
        return "BoundingBox: {:.2f} square pixels".format(self.area)


class EnclosingCircle:
    """ Associate a contour with its minimum enclosing circle (rotated to fit).

    Attributes:
        contour: ndarray
            The contour as computed by openCV.
        pos: int
            Arbitrary index that can be set to remember the position of this contour in a structure.
        circle: tuple
            The min enclosing circle of the contour, as computed by openCV.
        area: int
            The area of the enclosing circle.
    """

    def __init__(self, contour, pos):
        self.contour = contour
        self.pos = pos
        self.circle = cv2.minEnclosingCircle(contour)
        self.area = math.pi * self.circle[1]**2

    def __lt__(self, other):
        return self.area < other.area

    def __repr__(self):
        return "EnclosingCircle: {:.2f} square pixels".format(self.area)


class Segment:
    """ Helper class to store a segment (two 2d points), and offer util methods around it.
    Be careful when using it, since coordinates may be defined in openCV system, or numpy system across this project.

    Attributes:
        coords: tuple
            The first and last point of the segment, formatted as (x0, y0, x1, y1).
        theta: float
            The angle this segment forms with the horizontal line.
        offset: (int, int)
            Metadata that may be set from outside (ex: to re-integrate this Segment in a bigger image).
    """

    def __init__(self, coordinates):
        self.coords = coordinates
        self.theta = math.acos((coordinates[2] - coordinates[0]) / self.norm())
        self.offset = 0, 0

    def __getitem__(self, item):
        return self.coords[item]

    def p1(self):
        """ Return a tuple representing the first point of this Segment.
        """
        return self.coords[0], self.coords[1]

    def p2(self):
        """ Return a tuple representing the second point of this Segment.
        """
        return self.coords[2], self.coords[3]

    def norm(self):
        """ Return the euclidean (l2) norm of this Segment.
        """
        x2 = (self.coords[0] - self.coords[2]) ** 2
        y2 = (self.coords[1] - self.coords[3]) ** 2
        return math.sqrt(x2 + y2)

    def line_angle(self, other):
        """ Return the smallest angle between the line of direction "segment" and the line of direction "other".
        The return value is between 0 and pi / 2.
        """
        x0 = (self.coords[2] - self.coords[0]) / self.norm()
        y0 = (self.coords[3] - self.coords[1]) / self.norm()
        x1 = (other.coords[2] - other.coords[0]) / other.norm()
        y1 = (other.coords[3] - other.coords[1]) / other.norm()
        theta = math.acos(round(x0 * x1 + y0 * y1, 10))
        return theta if theta <= math.pi / 2 else math.pi - theta

    def intersection(self, other):
        """ Return the coordinates of the intersection of the infinite lines defined by "self" and "other".

        Args:
            other: Segment
        """
        x = (other[0] - self[0], other[1] - self[1])
        d1 = (self[2] - self[0], self[3] - self[1])
        d2 = (other[2] - other[0], other[3] - other[1])
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < sys.float_info.epsilon:
            return None
        else:
            t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
            return int(self[0] + t1 * d1[0]), int(self[1] + t1 * d1[1])

    def __str__(self):
        return "Seg({} + {}, {:.2f}rad)".format(self.coords, self.offset, self.theta)


class CyclicBuffer():
    """ Convenience class to handle buffering of multiple ndarrays (eg. accumulate them over time).

    Each buffered ndarray will be overwritten if the buffer index completes a cycle and new updates are submitted.

    The underlying structure is a numpy ndarray with one more dimension than the buffered ndarrays. The implementation
    of __getitem__ and __setitem__ magic methods aims at hiding that last dimension, since it is used as the buffer
    cycle index. However the self.buffer attribute is of course available, when an operation over different cycles must
    be performed.

    See doctest for some usage examples.

    Probably not intuitive (or optimal) in its design, but does the job for now.

    Attributes:
        size: int
            The size of the cycle: the number of ndarrays that can be buffered.
        buffer: ndarray
            The stucture containing the currently buffered values, as well as default values in the channels that
            haven't been taken yet.
        index: int
            The current channel in use for read (to get the last buffered array).

    To get / set values in the current cycle index, access the instance as a numpy array:

    >>> cb = CyclicBuffer((2, 2), 2, np.uint8, init=1)
    >>> print(cb[:])  # convenient access to currently buffered array
    [[1 1]
     [1 1]]
    >>> print(cb.buffer[:])  # underlying structure (contains 'size' buffered arrays)
    [[[1 1]
      [1 1]]
    <BLANKLINE>
     [[1 1]
      [1 1]]]
    >>> cb[0, 0] = 42  # update current array
    >>> cb[1, 1] = 42  # update current array (same array as above)
    >>> print(cb[:])
    [[42  1]
     [ 1 42]]
    >>> cb.increment()  # increment cycle index : point to the next buffered array
    >>> cb[0, 1] = 19  # update current array (now different from above)
    >>> print(cb[:])
    [[ 1 19]
     [ 1  1]]
    >>> print(cb.buffer[:])  # underlying structure : both updated arrays appear
    [[[42  1]
      [ 1 19]]
    <BLANKLINE>
     [[ 1  1]
      [42  1]]]
    >>> cb.increment()  # increment cycle index : a cycle has completed, so the buffer points on the first array again
    >>> cb[:] = 99
    >>> print(cb.buffer[:])
    [[[99  1]
      [99 19]]
    <BLANKLINE>
     [[99  1]
      [99  1]]]

    """

    def __init__(self, shape, size, dtype, init=None):
        self.size = size
        if type(shape) is int:
            shape = (shape, )
        self.buffer = np.zeros(shape + (size,), dtype=dtype)
        self.index = 0
        if init is not None:
            self.buffer[:] = init

    def __getitem__(self, item):
        if isinstance(item, tuple):
            assert len(item) < len(self.buffer.shape), \
                "Last dimension is used as implicit cycle marker, don't refer to it"
        return self.buffer.__getitem__(self._get_index(item))

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            assert len(item) < len(self.buffer.shape),\
                "Last dimension is used as implicit cycle marker, don't refer to it"
        self.buffer.__setitem__(self._get_index(item), value)

    def _get_index(self, item):
        """ Add one dimension to slices used to query this CycleBuffer.
        This dimension is used to point to the currently buffered channel (ndarray).

        Args:
            item: slice
                A slice as used when querying this CyclicBuffer.

        Returns buff_item: tuple
            The converted slice.
        """
        buff_item = item if isinstance(item, tuple) else (item, )
        while len(buff_item) < len(self.buffer.shape) - 1:
            # noinspection PyTypeChecker
            buff_item += (slice(None), )
        buff_item += (self.index % self.size, )
        return buff_item

    def replace(self, old, new):
        """ Find the next occurrence of "old", and replace it with "new". Start search right after the current index.
        """
        toreplace = old if isinstance(old, np.ndarray) else np.array([old])
        i = self.index
        for _ in range(self.size):
            i += 1
            if np.array_equal(self.buffer[..., i % self.size], toreplace):
                self.buffer[..., i % self.size] = new
                break

    def increment(self):
        self.index += 1

    def at_start(self) -> bool:
        """ Return true if self.index is pointing at the first position of the cycle.
        """
        return self.index % self.size == 0

    def at_end(self) -> bool:
        """ Return true if self.index is pointing at the last position of the cycle.
        """
        return (self.index + 1) % self.size == 0
