import math
import cv2
from bisect import insort
import numpy as np
from cam.draw import Segment, show

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


def hough(gray, prepare=True):
    """
    Calls the Canny function in order to call HoughLinesP function
    Returns all lines found.
    """
    # todo extract parameters in config file to ease tuning ?
    # todo remove "prepare" named parameter if not used
    if prepare:
        bw = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.Canny(bw, 10, 100)
    else:
        bw = gray

    threshold = 10
    #minlen = 3*min(bw.shape) / 4  # minimum length to accept a line
    minlen = min(bw.shape) / 2  # minimum length to accept a line
    maxgap = minlen / 12  # maximum gap for line merges (if probabilistic hough)

    lines = cv2.HoughLinesP(bw, 1, math.pi / 180, threshold, minLineLength=minlen, maxLineGap=maxgap)
    if lines is not None:
        return lines[0]
    else:
        return []


def find_segments(img):
    """
    Split the image into squares, call Hough on each square.
    Returns all segments found in two lists. The first contains the segments deemed "horizontal",
    and the other segment contains the rest, the "verticals". Both are ordered by intercept,
    their intersection with one of the middle line of the image.

    """

    hsegs = []
    vsegs = []
    chunks = list(split_sq(img, nbsplits=10))
    #chunks.extend(split_sq(img, nbsplits=10, offset=True))  # todo connect that if needing more segments
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        i += 1
        segments = hough(chunk.mat)
        for seg in segments:
            # translate segment coordinates to place it in global image
            seg[0] += chunk.x
            seg[1] += chunk.y
            seg[2] += chunk.x
            seg[3] += chunk.y
            segment = Segment(seg, img)
            if segment.horiz:
                insort(hsegs, segment)
            else:
                insort(vsegs, segment)

    return Grid(hsegs, vsegs, img)


if __name__ == '__main__':
    src = np.ones((5, 5), np.uint8)
    sub = src[1:4, 1:4]
    print(src)
    for x in range(sub.shape[1]):
        for y in range(sub.shape[0]):
            sub[x][y] = 0
    print(src)
    print zip(range(sub.shape[1]), range(sub.shape[0]))


class Grid:
    def __init__(self, hsegs, vsegs, img):
        assert isinstance(hsegs, list), "hsegs must be of type list."
        assert isinstance(vsegs, list), "vsegs must be of type list."
        self.hsegs = hsegs
        self.vsegs = vsegs
        self.img = img

    def __add__(self, other):
        assert isinstance(other, Grid), "can't add: other should be a grid."
        assert self.img.shape == other.img.shape, "images should have same shape when adding grids."
        hsegs = [seg for seg in self.hsegs + other.hsegs]
        vsegs = [seg for seg in self.vsegs + other.vsegs]
        hsegs.sort()
        vsegs.sort()
        return Grid(hsegs, vsegs, self.img)

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