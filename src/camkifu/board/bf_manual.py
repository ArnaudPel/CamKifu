import ntpath

import cv2
import numpy as np

from camkifu import board


__author__ = 'Arnaud Peloquin'


class BoardFinderManual(board.BoardFinder):
    """
    Let the user indicate the corners of the Goban manually, by clicking on each of them.

    """

    label = "Manual"

    def __init__(self, vmanager):
        """
        self.capture_pos -- used to lock the position of cvCapture when waiting for user to locate goban.
                            has no impact on live cam, but prevent unwanted frames consumption for files.

        """
        super().__init__(vmanager)
        self.manual_found = False
        self.capture_pos = None
        try:
            path = self.get_save_file_path()
            if path is not None:
                np_file = np.load(path)
                for p in np_file["location"]:
                    self.corners.submit(p)
                self.manual_found = True
        except IOError or TypeError:
            pass

    def _detect(self, frame):
        cv2.setMouseCallback(self._window_name(), self.onmouse)
        if not self.manual_found:
            self._lockpos()
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

    def onmouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.submit((x, y))
            if self.corners.is_ready():
                self.manual_found = True
                # noinspection PyProtectedMember
                path = self.get_save_file_path()
                if path is not None:
                    np.savez(path, location=self.corners._points)

    def get_save_file_path(self):
        try:
            from test.devconf import gobanloc_npz
            if type(self.vmanager.current_video) is str:
                fname = ntpath.basename(self.vmanager.current_video)
                return gobanloc_npz + fname + ".npz"
        except ImportError:
            return None

    def _window_name(self):
        return "Manual Grid Detection"
