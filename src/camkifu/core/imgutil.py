from bisect import insort
from math import sqrt, acos, pi, cos, sin
from sys import float_info, maxsize

from numpy import zeros, uint8, int32, ndarray, ones_like, arange, column_stack, flipud, vstack, array_equal,\
    array, sum as npsum, roll, where
from numpy.ma import minimum, around, sqrt as npsqrt
import cv2


__author__ = 'Arnaud Peloquin'


def connect_clusters(groups, dist):
    """
    Do one connectivity-based clustering pass: merge groups from which at least one couple of points
    is closer than 'dist' (the groups are separated by less than dist).

    This has been implemented quickly and is most likely not the best way to proceed.

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


def draw_circles(img, centers, color=(0, 0, 255), radius=5, thickness=1):
    for point in centers:
        if isinstance(point, ndarray) and point.shape == (1, 2):  # vertical points
            point = point.T
        x = point[0]
        y = point[1]
        point = (int(x), int(y))
        cv2.circle(img, point, radius, color=color, thickness=thickness)


def draw_lines(img, segments, color=(0, 255, 0)):
    thickness = 1 + 2 * _factor(img)
    for seg in segments:
        if isinstance(seg, Segment):
            p1 = (seg.coords[0] + seg.offset[0], seg.coords[1] + seg.offset[1])
            p2 = (seg.coords[2] + seg.offset[0], seg.coords[3] + seg.offset[1])
        else:
            if len(seg) == 4:
                p1 = (seg[0], seg[1])
                p2 = (seg[2], seg[3])
            elif len(seg) == 2:
                p1 = seg[0]
                p2 = seg[1]
            else:
                print("Unrecognized segment format: " + seg)
                continue
        try:
            cv2.line(img, p1, p2, color=color, thickness=thickness)
        except Exception as e:
            print(e)


def draw_str(dst, s, x=None, y=None):
    """
    Print a white string with a black shadow on the image. (Thank you opencv python samples)

    dst : the image where to print the string
    x : horizontal offset from left, in pixels
    y : vertical offset from top, in pixels
    s : the string to print

    """
    if x is None:
        x = int(dst.shape[0] / 2 - len(s) * 3.5)  # try to center horizontally depending on the string length
        y = int(dst.shape[1] / 2)
    # the shadow
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # the white text
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


windows = set()  # dev workaround to center windows the first time they are displayed only


def show(img, auto_down=True, name="Camkifu", loc=None):
    toshow = img
    if auto_down:
        f = _factor(img)
        if f:
            toshow = img.copy()
            name += " (downsized)"
            for i in range(f):
                toshow = cv2.pyrDown(toshow)

    if name not in windows:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if loc is not None:
            cv2.moveWindow(name, *loc)
        else:
            try:
                from test.devconf import screenw, screenh
                center = (screenw / 2, screenh / 2)
                cv2.moveWindow(name, max(0, int(center[0] - toshow.shape[0] / 2)), int(toshow.shape[1] / 2))
            except ImportError:
                pass
        windows.add(name)
    cv2.imshow(name, toshow)


def destroy_win(name):
    """
    Destroy the window having the provided name. Rise KeyError if this window had not been shown
    by the show() method implemented in this file.

    """
    windows.remove(name)
    cv2.destroyWindow(name)


def _factor(img):
    """
    Find how many times the image should be "pyrDown" to fit inside the screen.

    """
    f = 0
    try:
        from test.devconf import screenw, screenh
        imwidth = img.shape[1]
        imheight = img.shape[0]
        while screenw < imwidth or screenh < imheight:
            f += 1
            imwidth /= 2
            imheight /= 2
    except ImportError:
        pass
    return f


def saturate(img):
    """
    Convert the image to HSV, multiply both Saturation and Value by 1.5,
    and return corresponding enhanced BGR image (opencv defaults to BGR and not RGB).

    """
    maxsv = ones_like(img)
    maxsv[:, :, 1:3] *= 255
    saturated = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturated[:, :, 1:3] = minimum(maxsv[:, :, 1:3], saturated[:, :, 1:3] * 1.5)
    return cv2.cvtColor(saturated, cv2.COLOR_HSV2BGR)


def rgb_histo(img):
    """
    Code pasted from:
    http://opencvpython.blogspot.fr/2012/04/drawing-histogram-in-opencv-python.html

    """
    h = zeros((300, 256, 3))
    bins = arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = int32(around(hist_item))
        pts = column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)
    return flipud(h)


def segment_from_hough(hough_line, img_shape):
    """
    Create a Segment based on a line as returned by cv2.HoughLines()

    img_shape -- as img.shape, gives an order of magnitude for the segment length.
    """
    rho, theta = hough_line[0]
    a, b = cos(theta), sin(theta)
    x0, y0 = a * rho, b * rho
    extent = max(img_shape[0], img_shape[1])
    pt1 = int(x0 + extent * (-b)), int(y0 + extent * a)
    pt2 = int(x0 - extent * (-b)), int(y0 - extent * a)
    segment = Segment((pt1[0], pt1[1], pt2[0], pt2[1]))
    return segment


def cyclic_permute(cvhull):
    """
    Apply a cyclic permutation to the given hull points, so that the first point is the one showing at
    the upper left on the image. The overall sequence order of points is not modified.

    -- cvhull : a sequence of points (x, y).

    >>> cyclic_permute([(5, 367), (638, 364), (126, 96), (514, 92)])
    [(126, 96), (514, 92), (5, 367), (638, 364)]
    >>> cyclic_permute([(638, 364), (126, 96), (514, 92), (5, 367)])
    [(126, 96), (514, 92), (5, 367), (638, 364)]
    >>> cyclic_permute([(126, 96), (638, 364), (514, 92), (5, 367)])
    [(126, 96), (638, 364), (514, 92), (5, 367)]

    """
    hull = []
    idx = 0
    mind = maxsize
    for i in range(len(cvhull)):
        p = cvhull[i]
        dist = p[0] ** 2 + p[1] ** 2
        if dist < mind:
            mind = dist
            idx = i
    for i in range(idx, idx + len(cvhull)):
        p = cvhull[i % len(cvhull)]
        hull.append((p[0], p[1]))
    return hull


def get_ordered_hull(points):
    """
    Return the convex hull of the given points, so that:
    - the points are ordered clockwise
    - the first point is the closest to the origin (upper left corner)

    >>> get_ordered_hull([(5, 367), (638, 364), (126, 96), (514, 92)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]
    >>> get_ordered_hull([(5, 367), (126, 96), (638, 364), (514, 92)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]
    >>> get_ordered_hull([(5, 367), (126, 96), (514, 92), (638, 364)])
    [(126, 96), (514, 92), (638, 364), (5, 367)]

    """
    cvhull = cv2.convexHull(vstack(points))
    return cyclic_permute([x[0] for x in cvhull])


def _sort_contours(contours, wclass, area_bounds=None):
    sortedconts = []
    for i, cont in enumerate(contours):
        wrapped = wclass(cont, i)
        if area_bounds is None or area_bounds[0] < wrapped.area < area_bounds[1]:
            insort(sortedconts, wrapped)
    return sortedconts


def sort_contours_box(contours, area_bounds=None):
    """
    Sort contours by increasing bounding-box area.
    contours -- an iterable, as returned by cv2.findContours()

    Return -- a sorted list of BoundingBox objects. The position of each BB object corresponds to
    the contour's position in the provided iterable.

    """
    return _sort_contours(contours, BoundingBox, area_bounds=area_bounds)


def sort_contours_circle(contours, area_bounds=None):
    """
    Sort contours by increasing min enclosing-circle area.
    contours -- an iterable, as returned by cv2.findContours()

    Return -- a sorted list of EnclosingCircle objects. The position of each EC object corresponds to
    the contour's position in the provided iterable.

    """
    return _sort_contours(contours, EnclosingCircle, area_bounds=area_bounds)


def norm(p1, p2):
    """
    Return the euclidean norm of the vector defined by points p1 and p2.

    >>> "{:.6f}".format(norm((1, 1), (3, 7)))
    '6.324555'

    """
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def within_margin(p, box, margin):
    """
    Retrun True if the point "p" is located inside the rectangle "box", with respect of the provided safety margin.
    p  -- (x, y)
    box -- (x0, y0, x1, y1)
    margin -- an number

    """
    return box[0] + margin < p[0] < box[2] - margin and box[1] + margin < p[1] < box[3] - margin


def draw_contours_multicolor(img, contours):
    for i, contour in enumerate(contours):
        b = int(255 * (7 - i % 7) / 7)
        g = int(255 * (i % 5 + 1) / 5)
        r = int(255 * (3 - i % 3) / 3)
        cv2.drawContours(img, contours, i, (b, g, r))


def draw_closing_lines(img, contours):
    # todo decide whether or not this method should stay (implemented during "research" phase)
    for cont in contours:
        v1 = (roll(cont, -1, axis=0) - cont)
        v2 = (roll(cont, 1, axis=0) - cont)
        dotprod = npsum(v1 * v2, axis=2)
        norm1 = npsqrt(npsum(v1 ** 2, axis=2))
        norm2 = npsqrt(npsum(v2 ** 2, axis=2))
        cosinus = (dotprod / norm1) / norm2
        indexes = where(0.95 < cosinus)[0]
        if len(indexes) == 1:
            cv2.circle(img, tuple(cont[indexes[0], 0]), 3, (0, 255, 255))
        elif len(indexes) == 2:
            cv2.line(img, tuple(tuple(cont[indexes[0], 0])), tuple(cont[indexes[1], 0]), (0, 0, 255))
        else:
            for i in indexes:
                cv2.circle(img, tuple(cont[i, 0]), 3, (0, 0, 255))


class BoundingBox(object):
    """
    Wrapper of the bounding rectangle of a contour (rotated to fit), as computed by openCV.

    """
    def __init__(self, contour, pos):
        self.contour = contour
        self.pos = pos  # arbitrary index that can be set to remember position of this contour in a structure
        # self.box = cv2.boundingRect(contour)
        self.box = cv2.minAreaRect(contour)
        self.area = self.box[1][0]*self.box[1][1]

    def __lt__(self, other):
        return self.area < other.area

    def __repr__(self):
        return "BoundingBox: {:.2f} square pixels".format(self.area)


class EnclosingCircle(object):
    """
    Wrapper of the min enclosing circle of a contour, as computed by openCV.

    """
    def __init__(self, contour, pos):
        self.contour = contour
        self.pos = pos
        self.circle = cv2.minEnclosingCircle(contour)
        self.area = pi * self.circle[1]**2

    def __lt__(self, other):
        return self.area < other.area

    def __repr__(self):
        return "EnclosingCircle: {:.2f} square pixels".format(self.area)


class Segment:
    """
    Helper class to store a segment (two 2d points), and offer util methods around it.
    Coordinates storage format: (x0, y0, x1, y1).

    Note : in this project, the coordinates are usually defined in OpenCV coordinates system. Careful when using
    with numpy.

    """

    def __init__(self, coordinates):
        """
        seg -- (x0, y0, x1, y1)

        """
        self.coords = coordinates
        self.theta = acos((coordinates[2] - coordinates[0]) / self.norm())

        # metadata that may be set from outside (ex: to re-integrate this Segment in a bigger image).
        self.offset = 0, 0

    def __getitem__(self, item):
        return self.coords[item]

    def p1(self):
        return self.coords[0], self.coords[1]

    def p2(self):
        return self.coords[2], self.coords[3]

    def norm(self):
        """
        Return the euclidean (l2) norm of this segment.

        """
        x2 = (self.coords[0] - self.coords[2]) ** 2
        y2 = (self.coords[1] - self.coords[3]) ** 2
        return sqrt(x2 + y2)

    def line_angle(self, other):
        """
        Return the smallest angle between the line of direction "segment" and the line of direction "other"
        (return value between 0 and pi / 2).

        """
        x0 = (self.coords[2] - self.coords[0]) / self.norm()
        y0 = (self.coords[3] - self.coords[1]) / self.norm()
        x1 = (other.coords[2] - other.coords[0]) / other.norm()
        y1 = (other.coords[3] - other.coords[1]) / other.norm()
        theta = acos(round(x0 * x1 + y0 * y1, 10))
        return theta if theta <= pi / 2 else pi - theta

    def intersection(self, other):
        """
        Return the coordinates of the intersection of "self" with "other".

        """
        x = (other[0] - self[0], other[1] - self[1])
        d1 = (self[2] - self[0], self[3] - self[1])
        d2 = (other[2] - other[0], other[3] - other[1])
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < float_info.epsilon:
            return None
        else:
            t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
            return int(self[0] + t1 * d1[0]), int(self[1] + t1 * d1[1])

    def __str__(self):
        return "Seg({} + {}, {:.2f}rad)".format(self.coords, self.offset, self.theta)


class CyclicBuffer():
    """
    Convenience class to handle buffering of multi-dimensional arrays. Each buffered array will be overwritten
    if the buffer index completes a cycle and new updates are submitted.

    The underlying structure is a numpy array with one more dimension than the buffered arrays. The implementation of
    __getitem__ and __setitem__ magic methods aims at hiding that last dimension, since it is used as the buffer cycle
    index. However the self.buffer attribute is of course available, when an operation over different cycles must be
    performed.

    See doctest for some usage examples.

    Probably not optimal (and intuitive) in its design, but does the job for now.

    To get / set values in the current cycle index, access the instance as a numpy array:

    >>> cb = CyclicBuffer((2, 2), 2, uint8, init=1)
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
    >>> cb.increment()  # increment cycle index : point to the next buffered array
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
    >>> cb.increment()  # increment cycle index : a cycle has completed, so the buffer points on the first array again
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
        self.buffer = zeros(shape + (size,), dtype=dtype)
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
        buff_item = item if isinstance(item, tuple) else (item, )
        while len(buff_item) < len(self.buffer.shape) - 1:
            buff_item += (slice(None), )
        buff_item += (self.index % self.size, )
        return buff_item

    def replace(self, old, new):
        """
        Find the next occurrence of "old", and replace it with "new" (start search at current index and exclude it).

        """
        toreplace = old if isinstance(old, ndarray) else array([old])
        i = self.index
        for _ in range(self.size):
            i += 1
            if array_equal(self.buffer[..., i % self.size], toreplace):
                self.buffer[..., i % self.size] = new
                break

    def increment(self):
        self.index += 1

    def at_start(self) -> bool:
        """
        Return true if self.index is pointing at the first position of the cycle.

        """
        return self.index % self.size == 0

    def at_end(self) -> bool:
        """
        Return true if self.index is pointing at the last position of the cycle.

        """
        return (self.index + 1) % self.size == 0