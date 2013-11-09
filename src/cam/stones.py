from bisect import insort
import cv2
import numpy as np

from cam.board import ordered_hull
from cam.imgutil import draw_circles, show
from cam.video import VidProcessor
from config.guiconf import gsize, player_color

__author__ = 'Kohistan'


class StonesFinder(VidProcessor):
    """
    Save background data using sample(img)
    Perform background subtraction operations in order to detect stones.

    self.stones -- the matrix of stones found so far (0:None, 1:Black, 2:White)
    """

    def __init__(self, camera, rectifier, imqueue, transform, canonical_size):
        super(self.__class__, self).__init__(camera, rectifier, imqueue)
        self.bindings['s'] = self.reset
        self.observers = []

        self._transform = transform
        self._canonical_size = canonical_size

        # the 2 lines below would benefit from some sort of automation
        start = canonical_size / gsize / 2
        end = canonical_size - start
        self._grid = Grid([(start, start), (end, start), (end, end), (start, end)])

        self._background = np.zeros((gsize, gsize, 3), dtype=np.int16)
        self.dosample = True
        self.stones = np.zeros((gsize, gsize), dtype=np.uint8)
        self.lastpos = None

    def _doframe(self, frame):
        if self.undoflag:
            self.interrupt()  # go back to previous processing step
            self.reset()
        else:
            canon_img = cv2.warpPerspective(frame, self._transform, (self._canonical_size, self._canonical_size))
            if self.dosample:
                self.sample(canon_img)
                self.dosample = False
            else:
                self.detect(canon_img)
            self._drawgrid(canon_img)
            self._show(canon_img, name="Goban frame")

    def reset(self):
        self._background = self._background = np.zeros_like(self._background)
        self.stones = np.zeros_like(self.stones)
        self.lastpos = None
        self.dosample = True

    def _getzone(self, img, r, c):
        """
        r -- the intersection row index, numpy-like
        c -- the intersection column index, numpy-like

        """
        d, u = 1, 1  # the weights to give to vertical directions (down, up)
        proportions = [0.5, 1.0]  # needa be floats

        p = self._grid.pos[r][c]
        pbefore = self._grid.pos[r - 1][c - 1].copy()
        pafter = self._grid.pos[min(r + 1, gsize - 1)][min(c + 1, gsize - 1)].copy()
        if r == 0:
            pbefore[0] = -p[0]
        elif r == gsize - 1:
            pafter[0] = 2 * img.shape[0] - p[0] - 2
        if c == 0:
            pbefore[1] = -p[1]
        elif c == gsize - 1:
            pafter[1] = 2 * img.shape[1] - p[1] - 2

        rects = []
        points = []  # for debugging purposes. may be a good thing to clean that out
        # determine start and end point of the rectangle
        for i in range(2):
            w1 = proportions[i] / 2
            w2 = proportions[1-i] / 2
            start = (int(w1*pbefore[0] + (1-w1)*p[0]), int(w2*pbefore[1] + (1-w2)*p[1]))
            end = (int((1-w1)*p[0] + w1*pafter[0]), int((1-w2)*p[1] + w2*pafter[1]))
            rects.append(img[start[0]: end[0], start[1]: end[1]].copy())  # todo try to remove this copy()
            points.append((start[1], start[0], end[1], end[0]))

        return rects, points

    def sample(self, img):
        for x in range(gsize):
            for y in range(gsize):
                zones, points = self._getzone(img, x, y)
                #copy = img.copy()
                for i, zone in enumerate(zones):
                    for chan in range(3):
                        self._background[x][y][chan] = int(np.mean(zone[:, :, chan]))
                    #cv2.rectangle(copy, points[i][0:2], points[i][2:4], (255, 0, 0), thickness=-1)
                #show(copy, name="Sampling Zone")
                #if cv2.waitKey() == 113: raise SystemExit()
        print "Image at {0} set as background.".format(hex(id(img)))

    def detect(self, img):
        """
        Try to detect stones by comparing against neighbour colors.

        mask -- a matrix of shape (gsize, gsize) providing positions of already known stones, as follow.
                0: empty position
                1: white position
                2: black position

        """
        # todo: start to look for the new stone from the last known area (tenuki is rarely done at all moves)
        assert len(self._background) == gsize, "At least one sample must have been run to provide comparison data."
        pos = None
        val = 0
        #subtract = np.zeros_like(img)
        #deltas = []
        for x in range(gsize):
            for y in range(gsize):
                if not self.stones[x][y]:
                    bg = self._background[x][y]
                    current = np.zeros(3, dtype=np.int16)
                    zones, points = self._getzone(img, x, y)
                    delta = 0
                    for i, zone in enumerate(zones):
                        for chan in range(3):
                            current[chan] = int(np.mean(zone[:, :, chan]))
                        delta += compare(bg, current)
                    if delta < -200 or 400 < delta:
                        val = 1 if delta < 0 else 2
                        if pos is None:
                            pos = x, y
                        else:
                            print "dropped frame: StonesFinder (too many hits)"
                            return

                    #insort(deltas, delta)
                    #current -= bg
                    #current = np.absolute(current)
                    #color = (int(current[0]), int(current[1]), int(current[2]))
                    #for i in range(2):
                    #    cv2.rectangle(subtract, points[i][0:2], points[i][2:4], color, thickness=-1)
        #self._show(subtract, name="Subtract")
        #length = len(deltas)
        #print str(deltas[0:5]) + str(deltas[length-5:length])

        if pos is not None:
            if self.lastpos == pos:
                self.stones[pos] = val
                row = chr(97 + pos[1])
                col = chr(97 + pos[0])
                for obs in self.observers:
                    print "{0}[{1}{2}]".format(player_color[val], row, col)
                    obs.pipe("add", (row, col, player_color[val]))
            else:
                self.lastpos = pos

    def _drawgrid(self, img):
        if self._grid is not None:
            centers = []
            for i in range(19):
                for j in range(19):
                    centers.append(self._grid.pos[i][j])
                    draw_circles(img, centers)


def compare(background, current):
    """
    Return a distance between the two colors.
    background -- a vector of length 3
    current -- a vector of length 3

    """
    sign = 1 if np.sum(background) <= np.sum(current) else -1
    return sign*np.sum(np.absolute(current - background))


class Grid(object):
    """
    Store the location of each intersection of the goban.
    The aim of splitting that part in a separate class is to allow for a more automated and robust
    version to be developed.

    """

    def __init__(self, points):
        self.pos = np.zeros((gsize, gsize, 2), dtype=np.int16)
        hull = ordered_hull(points)
        assert len(hull) == 4, "The points expected here are the 4 corners of the grid."
        for i in range(gsize):
            xup = (hull[0][0] * (gsize - 1 - i) + hull[1][0] * i) / (gsize - 1)
            xdown = (hull[3][0] * (gsize - 1 - i) + hull[2][0] * i) / (gsize - 1)
            for j in range(gsize):
                self.pos[i][j][0] = (xup * (gsize - 1 - j) + xdown * j) / (gsize - 1)

                yleft = (hull[0][1] * (gsize - 1 - j) + hull[3][1] * j) / (gsize - 1)
                yright = (hull[1][1] * (gsize - 1 - j) + hull[2][1] * j) / (gsize - 1)
                self.pos[i][j][1] = (yleft * (gsize - 1 - i) + yright * i) / (gsize - 1)







# LEGACY CODE PROBABLY OUT OF DATE AND USE
#    self.perfectv = []
#    self.perfecth = []
#
#def _find_perfects(self, canon_img):
#    grid = find_segments(canon_img)
#    grid = runmerge(grid)
#    if len(self.perfecth) < 18:
#        for hseg in grid.hsegs:
#            if abs(hseg.slope) < 0.01:
#                insort(self.perfecth, hseg)
#    if len(self.perfectv) < 18:
#        for vseg in grid.vsegs:
#            if abs(vseg.slope) < 0.01:
#                insort(self.perfectv, vseg)
#
#def _process_perfects(self, canon_img):
#    anchors = [[], []]
#    perfects = (self.perfecth, self.perfectv)
#    for i in range(2):
#        rough_size = (float(canon_img.shape[1 - i]) / 19)
#        estimates = []
#        # to do: compare with more than one neighbour to leave "noisy couples" out ??
#        for j in range(len(perfects[i]) - 1):
#            gap = perfects[i][j + 1].intercept - perfects[i][j].intercept  # x1-x0 or y1-y0 depending on i
#            factor = gap / rough_size
#            if 0.90 < factor:
#                precision = (factor - floor(factor)) / round(factor)
#                if precision < 0.10 or 0.90 < precision:
#                    factor = round(factor)
#                    estimates.append(gap / factor)
#                    anchors[i].append(perfects[i][j])
#                    anchors[i].append(perfects[i][j + 1])
#
#        # attempt to correct gap using mean (possibly in a statistically wrong way..)
#        self.sizes[i] = np.mean(estimates)
#
#    # draw extrapolated positions
#    draw_lines(canon_img, self.perfecth, color=(255, 0, 0))
#    # draw detected positions
#    centers = []
#    for hanch in anchors[0]:
#        for vanch in anchors[1]:
#            intersect = hanch.intersection(vanch)
#            if intersect is not False:
#                centers.append(intersect)
#    draw_circles(canon_img, centers, color=(255, 0, 0))
#    draw_lines(canon_img, self.perfectv, color=(255, 0, 0))
#    draw_lines(canon_img, anchors[0])
#    draw_lines(canon_img, anchors[1])
#    show(canon_img, name="Grid Splitter")
#    self.reset()
#    cv2.waitKey()


























