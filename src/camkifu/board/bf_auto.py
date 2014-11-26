import cv2
from math import pi
from numpy import zeros, uint8

from camkifu.board.boardfinder import BoardFinder
from camkifu.core.imgutil import sort_contours_box, norm, get_ordered_hull, connect_clusters, segment_from_hough


__author__ = 'Arnaud Peloquin'


class BoardFinderAuto(BoardFinder):
    """
    Automatically detect the board location in an image. Implementation based on contours detection,
    with usage of median filtering. It is not able to detect the board when one or more stones are
    played on the first line, as those tend to break the contour.

    This implementation attempt is a work in progress.

    """

    label = "Automatic"

    def __init__(self, vmanager):
        super(BoardFinderAuto, self).__init__(vmanager)
        self.lines_accu = []
        self.groups_accu = []  # groups of points in the same image region
        self.passes = 0  # the number of frames processed since the beginning

    def _detect(self, frame):
        length_ref = min(frame.shape[0], frame.shape[1])  # a reference length linked to the image
        median = cv2.medianBlur(frame, 15)
        canny = cv2.Canny(median, 25, 75)
        # first parameter is the input image (it seems). appeared in opencv 3.0-alpha and is missing from the docs
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        sorted_boxes = sort_contours_box(contours)
        biggest = sorted_boxes[-1]
        frame_area = frame.shape[0] * frame.shape[1]
        found = False
        if frame_area / 3 < biggest.area:
            segments = self.find_lines(biggest.pos, contours, length_ref, median.shape)
            self.lines_accu.extend(segments)
            # accumulate data of 4 images before running board detection
            if not self.passes % 4:
                self.group_intersections(length_ref)
                while 4 < len(self.groups_accu):
                    prev_length = len(self.groups_accu)
                    connect_clusters(self.groups_accu, (length_ref / 50) ** 2)
                    if len(self.groups_accu) == prev_length:
                        break  # seems it won't get any better
                found = self.updt_corners(length_ref)

        if not self.passes % 4:
            self.corners.paint(median)
            self.metadata["Board  : {}"] = "found" if found else "searching"
            self._show(median)
        self.passes += 1
        return found

    @staticmethod
    def find_lines(pos, contours, length_ref, shape):
        """
        Use houghlines to find the sides of the contour (the goban hopefully).

        Another method could have been cv2.approxPolyDP(), but it fits the polygon points inside the contour, whereas
        what is needed here is a nicely fitted "tangent" for each side: the intersection points (corners) are outside
        the contour most likely (round corners).

        -- pos: the position to use in the list of contours
        -- contours: the list of contours
        -- length_ref: a reference used to parametrize thresholds
        -- the shape of the image in which contours have been found

        """
        ghost = zeros((shape[0], shape[1]), dtype=uint8)
        cv2.drawContours(ghost, contours, pos, (255, 255, 255), thickness=1)
        thresh = int(length_ref / 5)
        lines = cv2.HoughLines(ghost, 1, pi / 180, threshold=thresh)
        # todo increase threshold in a loop if there are too many lines found (or change other parameters)
        segments = []
        for line in lines:
            segments.append(segment_from_hough(line, ghost.shape))
        return segments

    def group_intersections(self, length_ref):
        """
        Store tangent intersections into rough groups (connectivity-based clustering idea).
        The objective is to obtain 4 groups of intersections, providing estimators of the corners
        location.

        """
        for s1 in self.lines_accu:
            for s2 in self.lines_accu:
                if pi / 3 < s1.line_angle(s2):
                    assigned = False
                    p0 = s1.intersection(s2)
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

    def updt_corners(self, length_ref):
        """
        If 4 groups have been found: compute the mean point of each group and update corners accordingly.
        In any case clear accumulation structures to prepare for next detection round.

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
                centers = get_ordered_hull(centers)  # so that the sides length can be calculated
                #   - check 1: each side must be of a certain length
                for i in range(len(centers)):
                    if norm(centers[i-1], centers[i]) < length_ref / 3:
                        found = False
                        break
                #   - check 2: at least one corner must have moved significantly (don't propagate motion due to approx)
                update = True if self.corners.hull is None else False
                if found and not update:
                    for i in range(nb_corners):
                        # hypothesis : both current and newly detected corners have been spacially sorted
                        if 5 < norm(centers[i], self.corners.hull[i]):
                            update = True
                            break
                if update:
                    self.corners.clear()
                    for pt in centers:
                        self.corners.submit(pt)
        self.metadata["Clusters : {}"].append(len(self.groups_accu))
        # self.metadata.append("lines accum : %d" % len(self.lines_accu))
        self.lines_accu.clear()
        self.groups_accu.clear()
        return found

    def _window_name(self):
        return "Board Finder Auto"