from math import floor
import cv2
import numpy as np
from bisect import insort

from cam.board import find_segments, runmerge, BoardFinder, ordered_hull
from cam.imgutil import show, draw_lines, VidProcessor, draw_circles, median_blur
from cam.calib import Rectifier

__author__ = 'Kohistan'


class StonesFinder(VidProcessor):
    def __init__(self, camera, rectifier):
        super(self.__class__, self).__init__(camera, rectifier)

        # these two guys below are expected to be set externally
        self.transform = None
        self.canonical_size = None

        self.perfectv = []
        self.perfecth = []
        self.sizes = [0, 0]

        self.grid = None

    def _find_perfects(self, canon_img):
        grid = find_segments(canon_img)
        grid = runmerge(grid)
        if len(self.perfecth) < 18:
            for hseg in grid.hsegs:
                if abs(hseg.slope) < 0.01:
                    insort(self.perfecth, hseg)
        if len(self.perfectv) < 18:
            for vseg in grid.vsegs:
                if abs(vseg.slope) < 0.01:
                    insort(self.perfectv, vseg)

    def _process_perfects(self, canon_img):
        anchors = [[], []]
        perfects = (self.perfecth, self.perfectv)
        for i in range(2):
            rough_size = (float(canon_img.shape[1 - i]) / 19)
            estimates = []
            # todo compare with more than one neighbour to leave "noisy couples" out.
            for j in range(len(perfects[i]) - 1):
                gap = perfects[i][j + 1].intercept - perfects[i][j].intercept  # x1-x0 or y1-y0 depending on i
                factor = gap / rough_size
                if 0.90 < factor:
                    precision = (factor - floor(factor)) / round(factor)
                    if precision < 0.10 or 0.90 < precision:
                        factor = round(factor)
                        estimates.append(gap / factor)
                        anchors[i].append(perfects[i][j])
                        anchors[i].append(perfects[i][j + 1])

            # attempt to correct gap using mean (possibly in a statistically wrong way..)
            self.sizes[i] = np.mean(estimates)

        # draw extrapolated positions
        draw_lines(canon_img, self.perfecth, color=(255, 0, 0))
        # draw detected positions
        centers = []
        for hanch in anchors[0]:
            for vanch in anchors[1]:
                intersect = hanch.intersection(vanch)
                if intersect is not False:
                    centers.append(intersect)
        draw_circles(canon_img, centers, color=(255, 0, 0))
        draw_lines(canon_img, self.perfectv, color=(255, 0, 0))
        draw_lines(canon_img, anchors[0])
        draw_lines(canon_img, anchors[1])
        show(canon_img, name="Grid Splitter")
        self.reset()
        cv2.waitKey()

    def _doframe(self, frame):
        canon_img = cv2.warpPerspective(frame, self.transform, (self.canonical_size, self.canonical_size))

        #if len(self.perfecth) < 18 or len(self.perfectv) < 18:
        #    self._find_perfects(canon_img)
        #else:
        #    self._process_perfects(canon_img)
        #draw_lines(canon_img, self.perfecth)
        #draw_lines(canon_img, self.perfectv)
        #show(canon_img, name="Grid Splitter")
        #if self.undo:
        #    self._interrupt()  # go back to previous processing step
        #    self.reset()

        #grid = find_segments(canon_img)
        #draw_lines(canon_img, grid.enumerate())
        #show(canon_img, name=

        #smooth = cv2.medianBlur(canon_img, 5)
        #smooth = cv2.bilateralFilter(canon_img, 9, 100, 100)
        #smooth = median_blur(canon_img, ksize=(5, 1))
        #show(smooth, name="Dev morphology")
        self._showgrid(canon_img)
        self.grid.sample(canon_img, np.ones((19, 19), dtype=np.bool))
        show(canon_img, name="Canoned")

    def _showgrid(self, img):
        start = self.canonical_size / 19 / 2
        end = self.canonical_size - start
        if self.grid is None:
            self.grid = Grid(19, points=[(start, start), (end, start), (end, end), (start, end)])
        centers = []
        for i in range(19):
            for j in range(19):
                centers.append(self.grid.mtx[i][j])
                #draw_circles(img, centers)

    def reset(self):
        self.perfecth = []
        self.perfectv = []


class Grid(object):
    """
    Stores the location of each intersection of the goban. This grid aims to be
    flexible, and provides methods to actualize positions.

    """

    def __init__(self, size, points=None):
        self.mtx = np.zeros((size, size, 2), dtype=np.int32)
        # a matrix of colors, 3 per position (i.e. 361*3)
        self.colors = []
        self.size = size

        if points is not None:
            hull = ordered_hull(points)
            assert len(hull) == 4, "The points expected here are the 4 corners of the grid."
            for i in range(size):
                xup = (hull[0][0] * (size - 1 - i) + hull[1][0] * i) / (size - 1)
                xdown = (hull[3][0] * (size - 1 - i) + hull[2][0] * i) / (size - 1)
                for j in range(size):
                    self.mtx[i][j][0] = (xup * (size - 1 - j) + xdown * j) / (size - 1)

                    yleft = (hull[0][1] * (size - 1 - j) + hull[3][1] * j) / (size - 1)
                    yright = (hull[1][1] * (size - 1 - j) + hull[2][1] * j) / (size - 1)
                    self.mtx[i][j][1] = (yleft * (size - 1 - i) + yright * i) / (size - 1)

    def sample(self, img, mask):
        for i in range(self.size):
            row = []
            self.colors.append(row)
            # build the matrix of mean colors
            for j in range(self.size):
                if mask[i][j]:
                    p = self.mtx[i][j]
                    pbefore = list(self.mtx[i - 1][j - 1])
                    pafter = list(self.mtx[min(i + 1, self.size - 1)][min(j + 1, self.size - 1)])

                    if i == 0:
                        pbefore[0] = -p[0]
                    elif i == self.size - 1:
                        pafter[0] = 2 * img.shape[0] - p[0] - 2
                    if j == 0:
                        pbefore[1] = -p[1]
                    elif j == self.size - 1:
                        pafter[1] = 2 * img.shape[1] - p[1] - 2

                    start = ((pbefore[0] + p[0]) / 2, (pbefore[1] + p[1]) / 2)
                    end = ((p[0] + pafter[0]) / 2, (p[1] + pafter[1]) / 2)
                    zone = img[start[0]: end[0], start[1]: end[1]]
                    color0 = int(np.mean(zone[:, :, 0]))
                    color1 = int(np.mean(zone[:, :, 1]))
                    color2 = int(np.mean(zone[:, :, 2]))
                    color = (color0, color1, color2)
                    row.append(color)

                    #imgc = img.copy()
                    #cv2.rectangle(imgc, start, end, color, thickness=-1)
                    #show(imgc, name="color means")
                    #if cv2.waitKey() == 113: return
                else:
                    row.append(None)
        for row in self.colors:
            print row
        print
        print


        #pbefore = self.mtx[max(i-imid, 0)][max(j-jmid, 0)]
        #pafter = self.mtx[max(i-imid, 0)][max(j-jmid, 0)]






























