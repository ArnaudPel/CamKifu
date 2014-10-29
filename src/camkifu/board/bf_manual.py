import os

import cv2
import numpy as np

from camkifu.board.boardfinder import BoardFinder
from test.devconf import gobanloc_npz


__author__ = 'Arnaud Peloquin'


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
            cv2.setMouseCallback(self._window_name(), self.onmouse)
            detected = False
        else:
            self._unlockpos()
            detected = True
        self.corners.paint(frame)
        self._show(frame)
        return detected

    def _lockpos(self):
        """
        Force capture to stay on the same image until the board has been located by user. It is necessary not to miss
        the first move(s). This code only impacts file processing, as live captures have no use for position ratio.

        """
        if self.capture_pos is None:
            self.capture_pos = self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO)
        else:
            self.vmanager.capt.set(cv2.CAP_PROP_POS_AVI_RATIO, self.capture_pos)

    def _unlockpos(self):
        """
        Free the movie run, as opposed to _lockpos().

        """
        self.capture_pos = None

    #noinspection PyUnusedLocal
    def onmouse(self, event, x, y, flag, param):
        # todo instead of forcing a particular order : make each click relocate the related point
        #   (the closest most likely, maybe also into account the cardinal location in the image)
        #   thus no need for cumbersome "undo" scheme
        if event == cv2.EVENT_LBUTTONDOWN and not self.corners.ready():
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

    def _window_name(self):
        return "Manual Grid Detection"
