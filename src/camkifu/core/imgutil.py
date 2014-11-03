from bisect import insort
from math import sqrt, acos, pi
from sys import float_info, maxsize

from numpy import zeros, int32, ndarray, ones_like, arange, column_stack, flipud, vstack
from numpy.ma import minimum, around
import cv2

from camkifu.config.cvconf import screenw, screenh


__author__ = 'Arnaud Peloquin'


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


def split_sq(img, nbsplits=5, offset=False):
    """
    Split the image in nbsplits * nbsplits squares and yield each Chunk.

    """
    for hchunk in split_h(img, nbsplits, offset=offset):
        for vchunk in split_v(hchunk.mat, nbsplits, offset=offset):
            yield Chunk(hchunk.x + vchunk.x, hchunk.y + vchunk.y, vchunk.mat, img)


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
            p1 = (seg.coords[0], seg.coords[1])
            p2 = (seg.coords[2], seg.coords[3])
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


def draw_str(dst, x, y, s):
    """
    Print a white string with a black shadow on the image. (Thank you opencv python samples)

    dst : the image where to print the string
    x : horizontal offset from left, in pixels
    y : vertical offset from top, in pixels
    s : the string to print

    """
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
            center = (screenw / 2, screenh / 2)
            cv2.moveWindow(name, max(0, int(center[0] - toshow.shape[0] / 2)), int(toshow.shape[1] / 2))
        windows.add(name)

    cv2.imshow(name, toshow)


def _factor(img):
    """
    Find how many times the image should be "pyrDown" to fit inside the screen.

    """
    f = 0
    imwidth = img.shape[1]
    imheight = img.shape[0]
    while screenw < imwidth or screenh < imheight:
        f += 1
        imwidth /= 2
        imheight /= 2
    return f


def saturate(img):
    """
    Convert the image to HSV, multiply both Saturation and Value by 1.5,
    and return corresponding enhanced RGB image.

    """
    maxsv = ones_like(img)
    maxsv[:, :, 1:3] *= 255
    saturated = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturated[:, :, 1:3] = minimum(maxsv[:, :, 1:3], saturated[:, :, 1:3] * 1.5)
    return cv2.cvtColor(saturated, cv2.COLOR_HSV2RGB)


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


def sort_contours(contours):
    """
    Sort contours by increasing bounding-box area.
    contours -- an iterable, as returned by cv2.findContours()

    Return -- a sorted list of Area objects. The position of each Area object corresponds to
    the contour's position in the provided iterable.

    """
    sortedconts = []
    for i, cont in enumerate(contours):
        insort(sortedconts, BoundingBox(cont, i))
    return sortedconts


def norm(p1, p2):
    """
    Return the euclidean norm of the vector defined by points p1 and p2.

    >>> "{:.6f}".format(norm((1, 1), (3, 7)))
    '6.324555'

    """
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


class BoundingBox(object):
    """
    Wrapper of the bounding rectangle of a contour (rotated to fit), as computed by openCV.

    """
    def __init__(self, contour, pos):
        self.contour = contour
        self.pos = pos  # arbitrary index that can be set to remember position of this contour in a structure
        # self.box = cv2.boundingRect(contour)
        self.box = cv2.minAreaRect(contour)
        self.box_area = self.box[1][0]*self.box[1][1]

    def __gt__(self, other):
        return other.box_area < self.box_area

    def __lt__(self, other):
        return self.box_area < other.box_area

    def __repr__(self):
        return "{0} pix2".format(self.box_area)


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


class Segment:
    """
    Segment that stores, on top of its coordinates:
    - an indicator of orientation (self.horiz, True for horizontal)
    - the intersection of this Segment with the corresponding middle line (horizontal for vertical and vice versa).
    - the slope of the segment (computation is orientation-dependent)

    Note: the concept of classifying segments by horizontal/vertical status is not good, which means this Segment
    implementation is not either. Classifying segments based on their relative orthogonality is more relevant.

    """
    def __init__(self, seg, img):

        xmid = img.shape[1] / 2
        ymid = img.shape[0] / 2
        xdiff = seg[2] - seg[0]
        ydiff = seg[3] - seg[1]

        self.coords = seg
        if abs(ydiff) < abs(xdiff):
            # horizontal segments (well, segments that are more horizontal than vertical)
            self.slope = float(ydiff) / xdiff
            self.intercept = self.slope * (xmid - seg[0]) + seg[1]
            self.horiz = True
        else:
            # vertical segments
            self.slope = float(xdiff) / ydiff
            self.intercept = self.slope * (ymid - seg[1]) + seg[0]
            self.horiz = False

    def p1(self):
        return self.coords[0], self.coords[1]

    def p2(self):
        return self.coords[2], self.coords[3]

    def __lt__(self, other):
        """
        Compare segments by their interception with middle line (not by their length !!)

        """
        return self.intercept < other.intercept

    def __gt__(self, other):
        """
        Compare segments by their interception with middle line (not by their length !!)

        """
        return self.intercept > other.intercept

    def __str__(self):
        return "Seg(intercept=" + str(self.intercept) + ")"

    def norm(self):
        x2 = (self.coords[0] - self.coords[2]) ** 2
        y2 = (self.coords[1] - self.coords[3]) ** 2
        return sqrt(x2 + y2)

    @staticmethod
    def lencmp(seg1, seg2):
        """
        Compare segments based on their L2-norm. The default comparison being on the intercepts,
        this one had to be external. see __lt__(self, other), __gt__(self, other)

        """
        a = len(seg1)
        b = len(seg2)
        # return cmp(a, b)
        return (a > b) - (a < b)  # quick fix made during 2to3

    def __getitem__(self, item):
        return self.coords[item]

    def angle(self, other):
        x0 = (self.coords[2] - self.coords[0]) / self.norm()
        y0 = (self.coords[3] - self.coords[1]) / self.norm()
        x1 = (other.coords[2] - other.coords[0]) / other.norm()
        y1 = (other.coords[3] - other.coords[1]) / other.norm()
        theta = acos(round(x0 * x1 + y0 * y1, 10))
        return theta if theta <= pi / 2 else pi - theta

    #noinspection PyNoneFunctionAssignment
    def intersection(self, other):
        x = (other[0] - self[0], other[1] - self[1])
        d1 = (self[2] - self[0], self[3] - self[1])
        d2 = (other[2] - other[0], other[3] - other[1])
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < float_info.epsilon:
            return None
        else:
            t1 = (x[0] * d2[1] - x[1] * d2[0]) / cross
            return int(self[0] + t1 * d1[0]), int(self[1] + t1 * d1[1])