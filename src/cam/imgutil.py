from math import sqrt
import math
import cv2
import numpy as np

from config.guiconf import screenw, screenh

__author__ = 'Kohistan'


def split_h(img, nbsplits=5, offset=False):
    """
    Split the image in "nbsplits" horizontal strips, and yields the Chunks.
    nbsplits -- the number of strips required.

    """

    for i in range(nbsplits):

        x0 = 0
        x1 = img.shape[1]

        strip_size = img.shape[0] / nbsplits
        if offset:
            offs = strip_size / 2
        else:
            offs = 0

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
        if offset:
            offs = strip_size / 2
        else:
            offs = 0
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
            # todo optimize loop if this is the way to go
            yield Chunk(hchunk.x + vchunk.x, hchunk.y + vchunk.y, vchunk.mat, img)


def draw_circles(img, centers, color=(0, 0, 255), radius=5, thickness=1):
    for point in centers:
        if isinstance(point, np.ndarray) and point.shape == (1, 2):  # bloody vertical points
            point = point.T
        x = point[0]
        y = point[1]
        point = (int(x), int(y))
        cv2.circle(img, point, radius, cv2.cv.CV_RGB(*color), thickness)


def draw_lines(img, segments, color=(0, 255, 0)):
    thickness = 1 + 2*_factor(img)
    for seg in segments:
        if isinstance(seg, Segment):
            p1 = (seg.coords[0], seg.coords[1])
            p2 = (seg.coords[2], seg.coords[3])
        else:
            p1 = (seg[0], seg[1])
            p2 = (seg[2], seg[3])
        colo = cv2.cv.CV_RGB(*color)
        cv2.line(img, p1, p2, colo, thickness=thickness)


def saturate(img):
    """
    Convert the image to HSV, multiply both Saturation and Value by 1.5,
    and return corresponding enhanced RGB image.

    """
    # todo add doctest for the 50% enhancement + respect of 255 max value
    maxsv = np.ones_like(img)
    maxsv[:, :, 1:3] *= 255
    saturated = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturated[:, :, 1:3] = np.minimum(maxsv[:, :, 1:3], saturated[:, :, 1:3] * 1.5)
    return cv2.cvtColor(saturated, cv2.COLOR_HSV2RGB)


def rgb_histo(img):
    """
    Code pasted from:
    http://opencvpython.blogspot.fr/2012/04/drawing-histogram-in-opencv-python.html

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

windows = set()  # todo improve or remove this dev workaround to center windows at startup only


def show(img, auto_down=True, name="Camkifu"):
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
        center = (screenw/2, screenh/2)
        cv2.moveWindow(name, max(0, center[0] - img.shape[0]/2), img.shape[1]/2)
        windows.add(name)

    cv2.imshow(name, toshow)


def _factor(img):
    """
    Find how many times the image should be "pyrDown" to fit inside the screen.

    """
#    screen size, automatic detection seems to be a pain so it is done manually.
    f = 0
    imwidth = img.shape[1]
    imheight = img.shape[0]
    while screenw < imwidth or screenh < imheight:
        f += 1
        imwidth /= 2
        imheight /= 2
    return f


def tohisto(mult_factor, values):

    """
    Take an iterable of float values, multiplies them by the factor,
    floor them and store occurrence count in a dict.
    """
    histo = {}
    for val in values:
        intv = int(math.floor(val * mult_factor))
        try:
            histo[intv] += 1
        except KeyError:
            histo[intv] = 1
    return histo


class Chunk:
    """
    A chunk of an image.

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
        dest[self.x:self.mat.shape[0] + self.x, self.y:self.mat.shape[1] + self.y] = self.mat


class Segment:

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

    def __lt__(self, other):
        return self.intercept < other.intercept

    def __gt__(self, other):
        return self.intercept > other.intercept

    def __str__(self):
        return "Seg(intercept=" + str(self.intercept) + ")"

    def __len__(self):
        x2 = (self.coords[0] - self.coords[2]) ** 2
        y2 = (self.coords[1] - self.coords[3]) ** 2
        return int(sqrt(x2 + y2))