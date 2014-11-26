from numpy import uint8, float32, reshape, unique, zeros, argmax, vectorize
from numpy.ma import absolute
import cv2
from camkifu.config.cvconf import canonical_size

from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, W, B, E

__author__ = 'Arnaud Peloquin'


class SfClustering(StonesFinder):

    label = "SF-Clustering"

    def __init__(self, vmanager):
        super().__init__(vmanager=vmanager)
        self.accu = None

    def _find(self, goban_img):
        gframe = goban_img
        if self.accu is None:
            self.accu = gframe.astype(float32)
        else:
            cv2.accumulateWeighted(gframe, self.accu, 0.2)
        if self.accu is not None and not self.total_f_processed % 3:
            ratios, centers = self.cluster_colors()
            self.check_pertinence(ratios, centers)

    def cluster_colors(self):
        """
        Objective : return for each intersection the percentage of B, W or E found based on pixel clustering
        according to their RGB (BGR) value.
        Computations based on the attribute self.accu

        """
        pixels = reshape(self.accu, (self.accu.shape[0] * self.accu.shape[1], 3))
        crit = (cv2.TERM_CRITERIA_EPS, 30, 3)
        retval, labels, centers = cv2.kmeans(pixels, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        if retval:
            # dev code to map the labels on an image to visualize the exact clustering result
            # centers_val = list(map(lambda x: int(sum(x) / 3), centers))  # wish I could vectorize the colors but.. failed
            # pixels = vectorize(lambda x: centers_val[x])(labels)
            # pixels = reshape(pixels.astype(uint8), (self.accu.shape[0], self.accu.shape[1]))
            # pixels *= self.getmask(pixels.shape)
            # self._show(pixels)
            # return None, None
            shape = self.accu.shape[0], self.accu.shape[1]
            labels = reshape(labels, shape)
            labels += 1  # don't leave any 0 before applying mask
            labels *= self.getmask(shape)
            # store each label percentage, over each intersection. Careful, they are not sorted, refer to "centers"
            ratios = zeros((gsize, gsize, 3), dtype=uint8)
            for x in range(gsize):
                for y in range(gsize):
                    # todo do +1 to the labels, and apply mask should help refining the percentages
                    zone, points = self._getzone(labels, x, y)
                    vals, counts = unique(zone, return_counts=True)
                    for i in range(len(vals)):
                        label = vals[i]
                        if 0 < label:
                            ratios[x][y][label - 1] = 100 * counts[i] / sum(counts)
            return ratios, centers

    def check_pertinence(self, ratios, centers):
        """
        Objective: check the result of a clustering method by using go-related logic. Eg having a filled mass of black
        on one side and the same big continuous mass of white on the other is not a game of Go.

        Also, more than 30 stones difference is a lot and should penalize the score.

        """
        if ratios is None:
            return
        c_vals = list(map(lambda x: int(sum(x) / 3), centers))  # grey level of centers
        c_colors = []
        for grey in c_vals:
            if grey == min(c_vals):
                c_colors.append(B)
            elif grey == max(c_vals):
                c_colors.append(W)
            else:
                c_colors.append(E)
        canvas = zeros((canonical_size, canonical_size), dtype=uint8)
        canvas[:] = 127
        # if an intersection is more than say 70% B or W, retain color. Otherwise assume it is empty.
        detected = zeros((gsize, gsize), dtype=object)
        for i in range(gsize):
            for j in range(gsize):
                max_k = argmax(ratios[i][j])
                detected[i][j] = c_colors[max_k]
                if c_colors[max_k] in (B, W):
                    y, x = self._posgrid.mtx[i][j]  # convert to opencv coords frame
                    cv2.circle(canvas, (x, y), 10, 0 if c_colors[max_k] is B else 255, thickness=-1)
        grid = self.search_intersections(self.accu.astype(uint8))
        conflicts = 0
        for i in range(gsize):
            for j in range(gsize):
                if sum(grid[i][j]) < 0 and detected[i][j] is not E:
                    conflicts += 1
        self.metadata["Conflict: {:.1f}%"] = 100 * conflicts / (gsize**2)
        # self.display_intersections(grid, canvas)
        self._posgrid.learn(absolute(grid))
        if conflicts < 7:
            moves = []
            for i in range(gsize):
                for j in range(gsize):
                    moves.append((detected[i][j], i, j))
            self.bulk_update(moves)
        self._show(canvas)

    def _learn(self):
        pass

    def _window_name(self):
        return SfClustering.label