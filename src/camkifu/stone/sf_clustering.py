import numpy as np
import cv2

import camkifu.stone
from golib.config.golib_conf import gsize, W, B, E


class SfClustering(camkifu.stone.StonesFinder):
    """ Stones finder implementation based on k-means pixel clustering.

    Has the ability to be used as a "standalone" finder, but as it is not designed to work in low stones-density
    context, it is put to better use when combined with other approaches. See SfMeta.

    Attributes:
        accu: ndarray
            Store a weighted sum of several successive Goban frames, in order to run the detection on an average image.
    """

    def __init__(self, vmanager):
        super().__init__(vmanager=vmanager)
        self.accu = None

    def _find(self, goban_img):
        """ Accumulate Goban frames, run stones detection every 3 images, and submit a bulk update accordingly.

        Note: SfMeta is using this class abilities, but it does via find_stones (thus bypassing this method).

        Args:
            goban_img: ndarray
                The already straightened up Goban image. It is expected to only contain the Goban pixels.
        """
        gframe = goban_img
        if self.accu is None:
            self.accu = gframe.astype(np.float32)
        else:
            cv2.accumulateWeighted(gframe, self.accu, 0.2)
        if not self.total_f_processed % 3:
            rs, re = 0, 19  # row start, row end
            cs, ce = 6, 13   # column start, column end
            stones = self.find_stones(self.accu, rs, re, cs, ce)
            if stones is not None:
                moves = []
                for i in range(gsize):
                    for j in range(gsize):
                        moves.append((stones[i][j], i, j))
                self.bulk_update(moves)

    def find_stones(self, img: np.ndarray, rs=0, re=gsize, cs=0, ce=gsize, **kwargs):
        """ The stones detection main algorithm, which is based on k-means pixel clustering.

        Note: the three colors (E, B, W) must be present in the image for this statistical method to work.

        Args:
            img: ndarray
                The Goban image.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.
            kwargs:
                Allowing for keyword args enables multiple find methods to be called indifferently. See SfMeta.

        Returns stones: ndarray
            A matrix containing the detected stones in the desired subregion of the image,
            or None if the result could not be trusted or something failed.
        """
        if img.dtype is not np.float32:
            img = img.astype(np.float32)
        ratios, centers = self.cluster_colors(img, rs=rs, re=re, cs=cs, ce=ce)
        stones = self.interpret_ratios(ratios, centers, r_start=rs, r_end=re, c_start=cs, c_end=ce)
        if not self.check_density(stones):
            return None  # don't trust this result
        return stones

    def cluster_colors(self, img: np.ndarray, rs=0, re=gsize, cs=0, ce=gsize) -> (np.ndarray, list):
        """ Compute for each intersection the percentage of B, W or E found by pixel color clustering (BGR value).
        Computations based on the attribute self.accu and cv2.kmeans (3-means).

        If a subregion only of the image is analysed (as per the arguments), the returned "ratios" array is still of
        global size (gsize * gsize), but the off-domain intersections are set to 1% goban and 0% other colors.

        Args:
            img: ndarray
                The Goban image.
            rs: int - inclusive
            re: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            cs: int - inclusive
            ce: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.

        Returns ratios, centers: ndarray, list
            The matrix of ratios. Its third dimension is used to store percentage of colors in each Goban intersection.
            The list of k-means centers (greyscale colors) that must be used to read the third dimension of the ratios
            matrix (which percentage is associated to which color).
        """
        x0, y0, _, _ = self.getrect(rs, cs)
        _, _, x1, y1 = self.getrect(re-1, ce-1)
        subimg = img[x0:x1, y0:y1]
        pixels = np.reshape(subimg, (subimg.shape[0] * subimg.shape[1], 3))
        crit = (cv2.TERM_CRITERIA_EPS, 15, 3)
        retval, labels, centers = cv2.kmeans(pixels, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)  # "attempts" a bit low ?
        if retval:
            # dev code to map the labels on an image to visualize the exact clustering result
            centers_val = list(map(lambda x: int(sum(x) / 3), centers))
            # pixels = vectorize(lambda x: centers_val[x])(labels)  # wish I could vectorize the colors but.. failed
            # pixels = reshape(pixels.astype(uint8), (subimg.shape[0], subimg.shape[1]))
            # pixels *= self.getmask(self.accu.shape[0:2])[x0:x1, y0:y1]
            # self._show(pixels)
            # return None, None
            shape = subimg.shape[0], subimg.shape[1]
            labels = np.reshape(labels, shape)
            labels += 1  # don't leave any 0 before applying mask
            labels *= self.getmask()[x0:x1, y0:y1].astype(labels.dtype)
            # store each label percentage, over each intersection. Careful, they are not sorted, refer to "centers"
            ratios = np.zeros((gsize, gsize, 3), dtype=np.uint8)
            ratios[:, :, centers_val.index(sorted(centers_val)[1])] = 1  # initialize 'E' channel to 1%
            for x in range(rs, re):
                for y in range(cs, ce):
                    a0, b0, a1, b1 = self.getrect(x, y)
                    # noinspection PyTupleAssignmentBalance
                    vals, counts = np.unique(labels[a0-x0:a1-x0, b0-y0:b1-y0], return_counts=True)
                    for i in range(len(vals)):
                        label = vals[i]
                        if 0 < label:
                            ratios[x][y][label - 1] = 100 * counts[i] / sum(counts)
            return ratios, centers

    def interpret_ratios(self, ratios, centers, r_start=0, r_end=gsize, c_start=0, c_end=gsize) -> np.ndarray:
        """ Interpret clustering results to retain one color per zone.

        Args:
            ratios: ndarray
                3D matrix storing for each zone the percentage of each "center" found in that zone.
            centers: list
                Gives the order in which percentages are stored in "ratios". Should be a 1D array of size 3,
                for colors B, W and E.
            r_start: int - inclusive
            r_end: int - exclusive
                Row start and end indexes. Can be used to restrain check to a subregion.
            c_start: int - inclusive
            c_end: int - exclusive
                Column start and end indexes. Can be used to restrain check to a subregion.

        Return stones: ndarray
            Indicate the color found for each zone (B, W or E).
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
        stones = np.zeros((gsize, gsize), dtype=object)
        stones[:] = E
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                max_k = np.argmax(ratios[i][j])
                stones[i][j] = c_colors[max_k]
        return stones

    def check_density(self, stones):
        """ Return True if the stones density is deemed enough for a 3-means clustering to make sense.
        """
        # noinspection PyTupleAssignmentBalance
        vals, counts = np.unique(stones, return_counts=True)
        # require at least two hits for each color
        if len(vals) < 3 or min(counts) < 2:
            return False
        return True

    def _learn(self):
        pass

    def _window_name(self):
        return SfClustering.__name__