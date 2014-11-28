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
            # ratios, centers = self.cluster_colors(row_end=12, col_end=12)
            ratios, centers = self.cluster_colors(row_start=7, row_end=12, col_start=7, col_end=12)
            conflicts, detected, grid = self.check_pertinence(ratios, centers)
            img = self.accu.astype(uint8)
            self.display_intersections(grid, img)
            # canvas = zeros((canonical_size, canonical_size), dtype=uint8)
            # canvas[:] = 127
            if conflicts < 7:  # todo make that relative to the observed zone (eg. < 2%, with one conflict authorized
                moves = []
                for i in range(gsize):
                    for j in range(gsize):
                        moves.append((detected[i][j], i, j))
                        # if detected[i][j] in (B, W):
                        #     y, x = self._posgrid.mtx[i][j]  # convert to opencv coords frame
                        #     cv2.circle(canvas, (x, y), 10, 0 if detected[i][j] is B else 255, thickness=-1)
                self.bulk_update(moves)
            # self._show(canvas)
            self._posgrid.learn(absolute(grid))  # to that at the end only, not to invalidate already computed values

    def cluster_colors(self, row_start=0, row_end=gsize, col_start=0, col_end=gsize):
        """
        Return for each analysed intersection the percentage of B, W or E found by pixel color clustering (BGR value).
        Computations based on the attribute self.accu and cv2.kmeans (3-means).

        If a subregion only of the image is analysed (as per the arguments), the returned "ratios" array is still of
        global size (gsize * gsize), but the off-domain intersections are set to 1% goban and 0% other colors.

        """
        x0, y0, _, _ = self._getrect(row_start, col_start)
        _, _, x1, y1 = self._getrect(row_end-1, col_end-1)
        subimg = self.accu[x0:x1, y0:y1]
        pixels = reshape(subimg, (subimg.shape[0] * subimg.shape[1], 3))
        crit = (cv2.TERM_CRITERIA_EPS, 30, 3)
        retval, labels, centers = cv2.kmeans(pixels, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        if retval:
            # dev code to map the labels on an image to visualize the exact clustering result
            centers_val = list(map(lambda x: int(sum(x) / 3), centers))  # wish I could vectorize the colors but.. failed
            # pixels = vectorize(lambda x: centers_val[x])(labels)
            # pixels = reshape(pixels.astype(uint8), (subimg.shape[0], subimg.shape[1]))
            # pixels *= self.getmask(self.accu.shape[0:2])[x0:x1, y0:y1]
            # self._show(pixels)
            # return None, None
            shape = subimg.shape[0], subimg.shape[1]
            labels = reshape(labels, shape)
            labels += 1  # don't leave any 0 before applying mask
            labels *= self.getmask(self.accu.shape[0:2])[x0:x1, y0:y1]
            # store each label percentage, over each intersection. Careful, they are not sorted, refer to "centers"
            ratios = zeros((gsize, gsize, 3), dtype=uint8)
            ratios[:, :, centers_val.index(sorted(centers_val)[1])] = 1  # initialize with goban
            for x in range(row_start, row_end):
                for y in range(col_start, col_end):
                    a0, b0, a1, b1 = self._getrect(x, y)
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(labels[a0-x0:a1-x0, b0-y0:b1-y0], return_counts=True)
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
        # if an intersection is more than say 70% B or W, retain color. Otherwise assume it is empty.
        detected = zeros((gsize, gsize), dtype=object)
        for i in range(gsize):
            for j in range(gsize):
                max_k = argmax(ratios[i][j])
                detected[i][j] = c_colors[max_k]
        grid = self.search_intersections(self.accu.astype(uint8))
        conflicts = 0
        for i in range(gsize):
            for j in range(gsize):
                if sum(grid[i][j]) < 0 and detected[i][j] is not E:
                    conflicts += 1
        self.metadata["Conflict: {:.1f}%"] = 100 * conflicts / (gsize**2)
        return conflicts, detected, grid

    def _learn(self):
        pass

    def _window_name(self):
        return SfClustering.label