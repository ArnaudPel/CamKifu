import ntpath

import cv2
import numpy as np

from camkifu import board


class BoardFinderManual(board.BoardFinder):
    """ Let the user indicate the corners of the Goban manually, by clicking on the image.

    This class also has the ability to save one location per input name, allowing to re-run files without having to
    manually detect each time. This feature is automatic, as long as the save directory is provided and exists.

    Attributes:
        manual_found: bool
            True if the 4 corners have been successfully clicked by the user.
        capture_pos: float
            The position of the video reading. Used to lock video progress in case of a file as long as the 4 corners
            of the image haven't been clicked, so that no information is lost while the user is marking the corners.
            Has no effect on live input.
    """

    def __init__(self, vmanager):
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
        """ Show a clickable image to the user, and interpret clicks as corners location.
        If reading a file, lock the position of the video progress until all 4 corners have been indicated.

        Args:
            The frame to process, which is actually sent to display for the user to do the work ;)

        Returns detected: bool
            True to indicate that the Goban has been located successfully (all 4 corners have been located).
        """
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
        """ Force capture to stay on the same image until the board has been fully located by user.

        This approach is necessary not to miss the first move(s) when reading a file.
        It only impacts file processing, as live captures have no use for position ratio.
        """
        if self.capture_pos is None:
            self.capture_pos = self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO)
        else:
            self.vmanager.capt.set(cv2.CAP_PROP_POS_AVI_RATIO, self.capture_pos)

    def _unlockpos(self):
        """ Let the movie run normally, as opposed to _lockpos().
        """
        self.capture_pos = None

    def onmouse(self, event, x, y, flag, param):
        """ Interpret mouse clicks as Goban corners locations. If all 4 have been found, try to save them to a file.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners.submit((x, y))
            if self.corners.is_ready():
                self.manual_found = True
                # noinspection PyProtectedMember
                path = self.get_save_file_path()
                if path is not None:
                    np.savez(path, location=self.corners._points)

    def get_save_file_path(self):
        """ Get the "locations save" directory from dev config, and create a save file path for the current input.
        Notes:
            - no location saved for live input (video must be an str to enable saving).
            - if dev config file is not present, ignore silently.

        Returns path: str
            The file path were to save the goban corners location.
        """
        try:
            from test.devconf import gobanloc_npz
            if type(self.vmanager.current_video) is str:
                fname = ntpath.basename(self.vmanager.current_video)
                return gobanloc_npz + fname + ".npz"
        except ImportError:
            return None

    def _window_name(self):
        return "Manual Grid Detection"
