import cv2
from cam.calib import Rectifier

from cam.board import find_segments, runmerge, BoardFinder
from cam.imgutil import show, draw_lines, VidProcessor

__author__ = 'Kohistan'


class StonesFinder(VidProcessor):

    def __init__(self, camera, rectifier):
        super(self.__class__, self).__init__(camera, rectifier)
        self.transform = None
        self.size = None
        self.perfectv = []
        self.perfecth = []

    def _doframe(self, frame):
        canon_img = cv2.warpPerspective(frame, self.transform, (self.size, self.size))

        if len(self.perfecth) < 6 or len(self.perfectv) < 6:
            grid = find_segments(canon_img)
            grid = runmerge(grid)
            if len(self.perfecth) < 6:
                for hseg in grid.hsegs:
                    if hseg.slope == 0:
                        self.perfecth.append(hseg)
            if len(self.perfectv) < 6:
                for vseg in grid.vsegs:
                    if vseg.slope == 0:
                        self.perfectv.append(vseg)

        draw_lines(canon_img, self.perfecth)
        draw_lines(canon_img, self.perfectv)
        show(canon_img, name="Perfect Lines")

        self.interrupt = self.undo