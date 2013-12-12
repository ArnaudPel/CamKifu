from bisect import insort
import math

import cv2
import numpy as np

from board.board1 import SegGrid, runmerge
from board.boardbase import BoardFinder
from core.imgutil import Segment, draw_lines, sort_conts


__author__ = 'Kohistan'


class BoardFinderAuto(BoardFinder):
    def __init__(self, vmanager, rect):
        super(BoardFinderAuto, self).__init__(vmanager, rect)

    def _detect(self, frame):

        median = cv2.medianBlur(frame, 15)
        canny = cv2.Canny(median, 25, 75)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        sortedconts = sort_conts(contours)
        ghost = np.zeros(frame.shape[0:2], dtype=np.uint8)
        for i in range(min(3, len(contours))):
            contid = sortedconts[-1 - i].pos
            cv2.drawContours(ghost, contours, contid, (255, 255, 255), thickness=1)

        threshold = 10
        minlen = min(*ghost.shape) / 3
        maxgap = minlen / 10
        lines = cv2.HoughLinesP(ghost, 1, math.pi / 180, threshold, minLineLength=minlen, maxLineGap=maxgap)

        found = False
        if lines is not None:
            draw_lines(median, lines[0])
            horiz = []
            vert = []
            for line in lines[0]:
                seg = Segment(line, frame)
                if seg.horiz:
                    insort(horiz, seg)
                else:
                    insort(vert, seg)
            grid = SegGrid(horiz, vert, frame)
            runmerge(grid)

            if len(grid.vsegs) == 2 and len(grid.hsegs) in (2, 3) \
                    and frame.shape[0] / 3 < abs(grid.vsegs[1].intercept - grid.vsegs[0].intercept) \
                    and frame.shape[1] / 3 < abs(grid.hsegs[1].intercept - grid.hsegs[0].intercept):
                found = True
                self.corners.clear()
                for i in range(2):
                    for j in range(2):
                        self.corners.add(grid.hsegs[i].intersection(grid.vsegs[j]))

        self.corners.paint(median)
        self._show(median, "Median")
        return found

    def perform_undo(self):
        super(BoardFinderAuto, self).perform_undo()
        self.corners.clear()