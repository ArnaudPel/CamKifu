import math
import cv2
import numpy as np
from cam.draw import Segment, draw_lines, _show, draw_circles
from cam.prepare import binarize
from cam.stats import tohisto
from gui.plot import plot_histo

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


def split_v(img, nbsplits=5):
    """
    Split the image in "nbsplits" vertical strips, and yields the Chunks.
    nbsplits -- the number of strips required.

    """
    for i in range(nbsplits):

        strip_size = img.shape[1] / nbsplits
        x0 = i * strip_size
        if (i + 1) < nbsplits:
            x1 = (i + 1) * strip_size
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
        for vchunk in split_v(hchunk.mat, nbsplits):
            # todo optimize loop if this is the way to go
            yield Chunk(hchunk.x + vchunk.x, hchunk.y + vchunk.y, vchunk.mat, img)


def hough(gray, prepare=True):
    """
    Calls the Canny function in order to call HoughLines2 function
    Returns all lines found. The image is binarized internally and can be provided in color.
    """
    # todo extract parameters in config file to ease tuning ?
    # todo remove "prepare" named parameter if not used
    if prepare:
        bw = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.Canny(bw, 10, 100)
    else:
        bw = gray

    threshold = 10
    minlen = min(bw.shape) / 2  # minimum length to accept a line
    maxgap = minlen / 12  # maximum gap for line merges (if probabilistic hough)

    lines = cv2.HoughLinesP(bw, 1, math.pi / 180, threshold, minLineLength=minlen, maxLineGap=maxgap)
    if lines is not None:
        return lines[0]
    else:
        return []


def find_squares(img):

    xmid = img.shape[1] / 2
    ymid = img.shape[0] / 2

    hlines = []
    vlines = []

    for chunk in split_sq(img, nbsplits=10):
        segments = hough(chunk.mat)
        for seg in segments:
            # translate segment coordinates to place it in global image
            x0 = chunk.x + seg[0]
            y0 = chunk.y + seg[1]
            x1 = chunk.x + seg[2]
            y1 = chunk.y + seg[3]

            # prepare some data that will be used in sorting and matching
            xdiff = x1 - x0
            ydiff = y1 - y0
            if xdiff != 0 and -1 < ydiff/xdiff < 1:
                slope = ydiff / xdiff
                yrank = slope * (xmid - x0) + y0
                hlines.append(Segment(((x0, y0), (x1, y1)), slope, yrank))
            else:
                inv_slope = xdiff / ydiff
                xrank = inv_slope * (ymid - y0) + x0
                vlines.append(Segment(((x0, y0), (x1, y1)), inv_slope, xrank))

    for chunk in split_sq(img, nbsplits=10, offset=True):
        segments = hough(chunk.mat)
        for seg in segments:
            # translate segment coordinates to place it in global image
            x0 = chunk.x + seg[0]
            y0 = chunk.y + seg[1]
            x1 = chunk.x + seg[2]
            y1 = chunk.y + seg[3]

            # prepare some data that will be used in sorting and matching
            xdiff = x1 - x0
            ydiff = y1 - y0
            if xdiff != 0 and -1 < ydiff/xdiff < 1:
                slope = ydiff / xdiff
                yrank = slope * (xmid - x0) + y0
                hlines.append(Segment(((x0, y0), (x1, y1)), slope, yrank))
            else:
                inv_slope = xdiff / ydiff
                xrank = inv_slope * (ymid - y0) + x0
                vlines.append(Segment(((x0, y0), (x1, y1)), inv_slope, xrank))

    # dev block
    #ghostgrid = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #ghostgrid = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2RGB)
    ghostgrid = np.zeros_like(img)
    centers = []
    for ln in hlines:
        centers.append((xmid, ln.intercept))
    for ln in vlines:
        centers.append((ln.intercept, ymid))
    draw_lines(ghostgrid, hlines, color=(255, 255, 255))
    draw_lines(ghostgrid, vlines, color=(255, 255, 255))
    #draw_circles(ghostgrid, centers)
    #draw_circles(ghostgrid, [(chunk.x, chunk.y)], color=(255, 0, 0))
    _show(ghostgrid, name="Ghost Grid")
    #cv2.waitKey()
    return ghostgrid


if __name__ == '__main__':
    src = np.ones((5, 5), np.uint8)
    sub = src[1:4, 1:4]
    print(src)
    for x in range(sub.shape[1]):
        for y in range(sub.shape[0]):
            sub[x][y] = 0
    print(src)
    print zip(range(sub.shape[1]), range(sub.shape[0]))


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