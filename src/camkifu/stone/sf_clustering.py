from numpy import uint8, float32, reshape, unique, zeros, argmax, vectorize, ndarray
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
            rs, re = 12, 19
            cs, ce = 6, 12
            ratios, centers = self.cluster_colors(self.accu)
            stones = self.interpret_ratios(ratios, centers)
            # ratios, centers = self.cluster_colors(self.accu, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            # stones = self.interpret_ratios(ratios, centers, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            grid = self.find_intersections(self.accu.astype(uint8))
            # lines, confirms = self.check_lines(stones, grid, r_start=rs, r_end=re, c_start=cs, c_end=ce)
            lines, confirms = self.check_lines(stones, grid)
            # canvas = zeros((canonical_size, canonical_size), dtype=uint8)
            # canvas[:] = 127
            if 4 < lines and lines - confirms < 5:
                moves = []
                for i in range(gsize):
                    for j in range(gsize):
                        moves.append((stones[i][j], i, j))
                        # if detected[i][j] in (B, W):
                        #     y, x = self._posgrid.mtx[i][j]  # convert to opencv coords frame
                        #     cv2.circle(canvas, (x, y), 10, 0 if detected[i][j] is B else 255, thickness=-1)
                self.bulk_update(moves)
            else:
                self.metadata["Too few confirms, skipped frame"] = None
            img = self.accu.astype(uint8)
            self.display_intersections(grid, img)
            # self._show(canvas)
            self._posgrid.learn(absolute(grid))  # do that at the end only, not to invalidate already computed values

    def cluster_colors(self, img: ndarray, r_start=0, r_end=gsize, c_start=0, c_end=gsize) -> (ndarray, list):
        """
        Return for each analysed intersection the percentage of B, W or E found by pixel color clustering (BGR value).
        Computations based on the attribute self.accu and cv2.kmeans (3-means).

        If a subregion only of the image is analysed (as per the arguments), the returned "ratios" array is still of
        global size (gsize * gsize), but the off-domain intersections are set to 1% goban and 0% other colors.

        """
        x0, y0, _, _ = self._getrect(r_start, c_start)
        _, _, x1, y1 = self._getrect(r_end-1, c_end-1)
        subimg = img[x0:x1, y0:y1]
        pixels = reshape(subimg, (subimg.shape[0] * subimg.shape[1], 3))
        crit = (cv2.TERM_CRITERIA_EPS, 15, 3)
        retval, labels, centers = cv2.kmeans(pixels, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)  # "attempts" a bit low ?
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
            labels *= self.getmask(img.shape[0:2])[x0:x1, y0:y1]
            # store each label percentage, over each intersection. Careful, they are not sorted, refer to "centers"
            ratios = zeros((gsize, gsize, 3), dtype=uint8)
            ratios[:, :, centers_val.index(sorted(centers_val)[1])] = 1  # initialize with goban
            for x in range(r_start, r_end):
                for y in range(c_start, c_end):
                    a0, b0, a1, b1 = self._getrect(x, y)
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = unique(labels[a0-x0:a1-x0, b0-y0:b1-y0], return_counts=True)
                    for i in range(len(vals)):
                        label = vals[i]
                        if 0 < label:
                            ratios[x][y][label - 1] = 100 * counts[i] / sum(counts)
            return ratios, centers

    def interpret_ratios(self, ratios, centers, r_start=0, r_end=gsize, c_start=0, c_end=gsize) -> ndarray:
        """
        Interpret clustering results to retain one color per zone.
        ratios -- a 3D matrix storing for each zone the percentage of each "center" found in that zone.
        centers -- gives the order in which percentages are stored in "ratios". Should be a 1D array of size 3,
                   for colors B, W and E.

        Return "stones", a 2D array containing for each zone either B, W or E.

        """
        # 1. map centers to colors
        assert len(centers) == 3
        c_vals = list(map(lambda x: int(sum(x) / 3), centers))  # grey level of centers
        c_colors = []
        for grey in c_vals:
            if grey == min(c_vals):
                c_colors.append(B)
            elif grey == max(c_vals):
                c_colors.append(W)
            else:
                c_colors.append(E)
        # 2. set each intersection's color
        stones = zeros((gsize, gsize), dtype=object)
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                max_k = argmax(ratios[i][j])
                stones[i][j] = c_colors[max_k]
        return stones

    def check_lines(self, stones: ndarray, grid: ndarray, r_start=0, r_end=gsize, c_start=0, c_end=gsize):
        """
        Check that the provided "stones" 2D array is coherent with lines detection in the image: no line should
        be found in zones where a stone has been detected.

        A confirmation is counted for the zone if it is empty (E) and at least one line has also been detected
        in that zone.

        stones -- a 2D array that can store the objects, used to record the stones found. It is created if not provided.
        grid -- a 3D (or 2D) array that has negative values where lines have been found.

        Return
        lines -- the number of intersections where lines have been detected (based on the provided grid)
        confirms -- the number of empty intersections where lines have been detected, meaning the

        """
        lines = 0     # the number of intersections where lines have been detected
        confirms = 0  # the number of empty intersections where lines have been detected
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                if sum(grid[i][j]) < 0:
                    lines += 1
                    if stones[i][j] is E:
                        confirms += 1
        return lines, confirms

    def _learn(self):
        pass

    def _window_name(self):
        return SfClustering.label