import os

import cv2
from cv import CV_CAP_PROP_POS_AVI_RATIO as POS_RATIO
import numpy as np
from board.boardfinder import BoardFinder

from test.devconf import gobanloc_npz


__author__ = 'Arnaud Peloquin'


if __name__ == '__main__':
    src = np.ones((5, 5), np.uint8)
    sub = src[1:4, 1:4]
    print(src)
    for x in range(sub.shape[1]):
        for y in range(sub.shape[0]):
            sub[x][y] = 0
    print(src)
    print zip(range(sub.shape[1]), range(sub.shape[0]))


class BoardFinderManual(BoardFinder):
    """
    Let the user indicate the corners of the Goban manually, by clicking on each of them.
    Press 'z' to undo one click.

    """

    label = "Manual"

    def __init__(self, vmanager):
        """
        self.capture_pos -- used to lock the position of cvCapture when waiting for user to locate goban.
                            has no impact on live cam, but prevent unwanted frames consumption for files.

        """
        super(BoardFinderManual, self).__init__(vmanager)
        self.windowname = "Manual Grid Detection"
        self.manual_found = False
        self.capture_pos = None
        try:
            np_file = np.load(gobanloc_npz)
            for p in np_file["location"]:
                self.corners.add(p)
            self.manual_found = True
        except IOError or TypeError:
            pass

    def _detect(self, frame):
        if self.undoflag:
            self.perform_undo()
        if not self.manual_found:
            self._lockpos()
            detected = False
        else:
            self._standby()
            detected = True
        self.corners.paint(frame)
        self._show(frame, name=self.windowname)
        return detected

    def _lockpos(self):
        if self.capture_pos is None:
            self.capture_pos = self.vmanager.capt.get(POS_RATIO)
        else:
            self.vmanager.capt.set(POS_RATIO, self.capture_pos)
        cv2.setMouseCallback(self.windowname, self.onmouse)

    def _standby(self):
        self.capture_pos = None

    #noinspection PyUnusedLocal
    def onmouse(self, event, x, y, flag, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not self.corners.ready():
            self.corners.add((x, y))
            if self.corners.ready():
                self.manual_found = True
                np.savez(gobanloc_npz, location=self.corners._points)

    def perform_undo(self):
        super(BoardFinderManual, self).perform_undo()
        self.manual_found = False
        self.corners.pop()
        try:
            os.remove(gobanloc_npz)
        except OSError:
            pass