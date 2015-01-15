import time
import math

import cv2
import numpy as np

from camkifu import board
from camkifu.core import imgutil


__author__ = 'Arnaud Peloquin'


class BoardFinderAuto(board.BoardFinder):
    """ Automatically detect the board location in an image.

    Implementation based on contours detection, with usage of median filtering. It is not able to detect the board
    when one or more stones are played on the first line, as those tend to break the contour.

    This implementation attempt is a work in progress.

    Attributes:
        lines_accu: list
            Accumulator of lines detected as probable Goban sides (any of the 4 sides).
        groups_accu: list
            Accumulator of points that are candidates identifiers of Goban corners. They are obtained by
            intersecting the accumulated lines described above. They are grouped together when being close enough to
            each other, and a positive detection can be triggered if 4 groups stand out.
        auto_refresh: int
            The number of seconds to wait after the last positive detection before searching again.
        last_positive: int
            The last time the board has been detected.
    """

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.lines_accu = []
        self.groups_accu = []

        self.auto_refresh = 10
        self.last_positive = -1.0

    def _doframe(self, frame):
        """ Decoration of super()._doframe() to add a sleeping period after a positive board detection.
        """
        last_positive = time.time() - self.last_positive
        if self.auto_refresh < last_positive:
            super()._doframe(frame)
        else:
            self.corners.paint(frame)
            self.metadata["Last detection {}s ago"] = int(last_positive)
            self._show(frame)

    def _detect(self, frame):
        """ Look for a goban in the provided frame. It is assumed that this method will be called in a loop.

        The idea followed here is to search a large rectangular object in the center of the image. Since the
        Goban lines are not specifically searched, a median blur is applied to try to make the Goban edges sharper.

        Then a contours search is run, and the first few biggest contours are analysed to find long lines.

        Those lines are accumulated over a few frames (hence the important assumption that this method is being called
        in a loop). They are periodically intersected to get candidate points for Goban corners. Finally, the points
        are grouped to get a mean position of 4 corners. If more or less than 4 groups of candidates are obtained,
        the detection is deemed unsuccessful and cleared for another pass.

        Args:
            frame: ndarray
                The image to search.

        Returns detected: bool
            True to indicate that the Goban has been located successfully (all 4 corners have been located).
        """
        length_ref = min(frame.shape[0], frame.shape[1])  # a reference length linked to the image
        median = cv2.medianBlur(frame, 15)
        canny = cv2.Canny(median, 25, 75)
        # first parameter is the input image (it seems). appeared in opencv 3.0-alpha and is missing from the docs
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        sorted_boxes = imgutil.sort_contours_box(contours)
        biggest = sorted_boxes[-1]
        frame_area = frame.shape[0] * frame.shape[1]
        found = False
        if frame_area / 3 < biggest.area:
            positions = [x.pos for x in sorted_boxes[-3:]]  # look for lines in the 3 "biggest" contours
            segments = self.find_lines(positions, contours, length_ref, median.shape)
            self.lines_accu.extend(segments)
            # accumulate data of 4 images before running board detection
            if not self.total_f_processed % 4:
                self.group_intersections(frame.shape)  # fill self.groups_accu
                while 4 < len(self.groups_accu):
                    prev_length = len(self.groups_accu)
                    imgutil.connect_clusters(self.groups_accu, (length_ref / 50) ** 2)
                    if len(self.groups_accu) == prev_length:
                        break  # seems it won't get any better
                found = self.updt_corners(length_ref)

        if not self.total_f_processed % 4:
            self.corners.paint(median)
            self.metadata["Board  : {}"] = "found" if found else "searching"
            self._show(median)
        if found:
            self.last_positive = time.time()
        return found

    # noinspection PyMethodMayBeStatic
    def find_lines(self, posititons, contours, length_ref, shape):
        """ Use houghlines to find big lines in the contours.

        Another approach could have used cv2.approxPolyDP(), but it fits the polygon points inside the contour, whereas
        what is needed here is a nicely fitted "tangent" for each side: the intersection points (corners) are outside
        the contour most likely (round corners).

        Args:
            positions: ints
                The indexes of elements of interest in 'contours'.
            contours: list
                A list of contours.
            length_ref: int
                A length reference used to parametrize thresholds.
            shape: ints
                The shape of the image in which contours have been found

        Returns segments: list
            The lines found in the desired contours.
        """
        ghost = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        # colors = ((0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255))
        # i = 0
        for pos in posititons:
            cv2.drawContours(ghost, contours, pos, 255, thickness=1)
            # i += 1
        # self._show(ghost)
        thresh = int(length_ref / 5)
        lines = cv2.HoughLines(ghost, 1, math.pi / 180, threshold=thresh)
        segments = []
        for line in lines:
            segments.append(imgutil.segment_from_hough(line, ghost.shape))
        # canvas = zeros(ghost.shape, dtype=uint8)
        # draw_lines(canvas, segments, color=255)
        # self.metadata["Nb lines: {}"] = len(segments)
        # self._show(canvas)
        return segments

    def group_intersections(self, shape):
        """ Store lines intersections points into rough groups (connectivity-based clustering).

        The objective is to obtain 4 groups of intersections, as estimators of the corners location. Those groups
        are updated in the 'groups_accu' attribute, hence the absence of return value.
        """
        length_ref = min(shape[0], shape[1])  # a reference length linked to the image
        sorted_lines = sorted(self.lines_accu, key=lambda x: x.theta)
        for s1 in sorted_lines:
            for s2 in reversed(sorted_lines):
                if math.pi / 3 < s1.line_angle(s2):
                    assigned = False
                    p0 = s1.intersection(s2)
                    margin = -length_ref / 15  # allow intersections to be outside of the image a bit
                    if imgutil.within_margin(p0, (0, 0, shape[1], shape[0]), margin):
                        for g in self.groups_accu:
                            for p1 in g:
                                # if the two points are close enough, group them
                                if (p0[0] - p1[0]) ** 2 + (p0[0] - p1[0]) ** 2 < (length_ref / 80) ** 2:
                                    g.append(p0)
                                    assigned = True
                                    break
                            if assigned:
                                break
                        if not assigned:
                            # create a new group for each lonely point, it may be joined with others later
                            self.groups_accu.append([p0])
                else:
                    # s2 is "too parallel" with s1: all possibly interesting intersections have been processed for s1.
                    break

    def updt_corners(self, length_ref):
        """ Compute the mean of each points group and update corners if 4 groups have been found.
        In any case clear accumulation structures to prepare for next detection round.

        Returns found: bool
            True if 4 corners have been detected, False otherwise.
        """
        found = False
        nb_corners = 4
        if len(self.groups_accu) == nb_corners:
                # step 1 : compute center of mass of each group
                centers = []
                for group in self.groups_accu:
                    x, y = 0, 0
                    for pt in group:
                        x += pt[0]
                        y += pt[1]
                    centers.append((int(x / len(group)), int(y / len(group))))
                # step 2 : run some final checks on the resulting corners
                found = True
                centers = imgutil.get_ordered_hull(centers)  # so that the sides length can be calculated
                #   - check 1: each side must be of a certain length
                for i in range(len(centers)):
                    if imgutil.norm(centers[i-1], centers[i]) < length_ref / 3:
                        found = False
                        break
                #   - check 2: at least one corner must have moved significantly (don't propagate motion due to approx)
                update = True if self.corners.hull is None else False
                if found and not update:
                    for i in range(nb_corners):
                        # hypothesis : both current and newly detected corners have been spacially sorted
                        if 5 < imgutil.norm(centers[i], self.corners.hull[i]):
                            update = True
                            break
                if update:
                    self.corners.clear()
                    for pt in centers:
                        self.corners.submit(pt)
        self.metadata["Clusters : {}"].append(len(self.groups_accu))
        self.metadata["Line intersections: {}"] = sum([len(g) for g in self.groups_accu])
        # self.metadata.append("lines accum : %d" % len(self.lines_accu))
        self.lines_accu.clear()
        self.groups_accu.clear()
        return found

    def _window_name(self):
        return "Board Finder Auto"