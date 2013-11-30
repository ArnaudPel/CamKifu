import numpy as np

from stone.stonesbase import StonesFinder, compare
from golib_conf import gsize, player_color

__author__ = 'Kohistan'


class BackgroundSub(StonesFinder):
    """
    Save background data using sample(img)
    Perform background subtraction operations in order to detect stones.

    self.stones -- the matrix of stones found so far (0:None, 1:Black, 2:White)
    """

    def __init__(self, camera, rectifier, imqueue, transform, canonical_size):
        super(BackgroundSub, self).__init__(camera, rectifier, imqueue, transform, canonical_size)
        self.bindings['s'] = self.reset

        self._background = np.zeros((gsize, gsize, 3), dtype=np.int16)
        self.dosample = True
        self.lastpos = None

    def _find(self, goban_img):
        if self.dosample:
            self.sample(goban_img)
            self.dosample = False
        else:
            self.detect(goban_img)
        self._drawgrid(goban_img)
        self._show(goban_img, name="Goban frame")

    def reset(self):
        super(BackgroundSub, self).reset()
        self._background = self._background = np.zeros_like(self._background)
        self.lastpos = None
        self.dosample = True

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
                    obs.pipe("append", (player_color[val], row, col))
            else:
                self.lastpos = pos





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


























