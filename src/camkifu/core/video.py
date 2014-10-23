from queue import Full
from threading import current_thread
from time import sleep, time

import cv2

from camkifu.core.imgutil import show, draw_str


__author__ = 'Arnaud Peloquin'


class VidProcessor(object):
    """
    Class meant to be extended by implementations of video processing.
    By default the execute() method will keep running, interruption must be asked.

    """

    def __init__(self, vmanager):
        """
        self.frame_period -- (float) the minimum number of seconds between 2 iterations of frame processing

        """
        self.vmanager = vmanager
        self.own_images = {}

        self.frame_period = 0.2  # shortest period between two processings: put thread to sleep when possible
        self.last_read = 0.0  # gives the instant of the last image processing start (in seconds).

        self.undoflag = False
        self._interruptflag = False
        self.pausedflag = False

        self.bindings = {'p': self.pause, 'q': self.interrupt, 'z': self.undo}
        self.key = None
        self.latency = 0.0

    def execute(self):
        """
        Execute the video processing loop. Run until self.interrupt() is called, or if the
        last frame has been reached when reading a file.

        In order to save some CPU, the loop sleeps between two iterations if they are processed
        faster than self.frame_period. Set this value to 0 to run flat out.

        """
        end = self.vmanager.controller.bounds[1]
        self._interruptflag = False
        while not self._interruptflag and (self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO) < end):
            self._checkpause()
            start = time()
            if self.frame_period < start - self.last_read:
                self.last_read = start
                ret, frame = self.vmanager.capt.read()
                if ret:
                    self._doframe(frame)
                    self.checkkey()
                    self.latency = time() - start
                else:
                    print("Could not read camera for {0}.".format(str(type(self))))
                    self.latency = 0.0
                    sleep(5)
            else:
                sleep(self.frame_period / 10)  # precision doesn't really matter here
        self._destroy_windows()
        self.vmanager.confirm_stop(self)

    def _doframe(self, frame):
        """
        Image-processing algorithm may be implemented under that extension point.

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def interrupt(self):
        """
        Stop the video processing loop.

        """
        self._interruptflag = True

    def pause(self, dopause=None):
        """
        Pause the video processing loop if "dopause" is True, otherwise resume loop.

        """
        if dopause is not None:
            # set provided value
            self.pausedflag = dopause
        else:
            # no value provided, interpret as a toggle
            self.pausedflag = not self.pausedflag

    def _checkpause(self):
        """
        Multi-threaded env: will sleep thread as long as self.pausedflag is True.
        Single-threaded env: will keep calling cv.waitKey until a valid command is pressed.

        """
        if self.vmanager.imqueue is not None:  # supposedly a multi-threaded env
            if self.pausedflag:
                while self.pausedflag:
                    # re-show last images, in order to re-activate the waitkey()
                    for name, img in self.own_images.items():
                        self._show(img, name=name)
                    sleep(0.1)
                    self.checkkey()
        else:  # supposedly in single-threaded dev mode
            if self.pausedflag:
                key = None
                while True:
                    # repeating the same key resumes processing. other keys are executed as if nothing happened
                    try:
                        key = chr(cv2.waitKey(500))
                        if key in self.bindings:
                            break
                    except ValueError:
                        pass
                if (key is not None) and key != 'p':
                    command = self.bindings[key]
                    if command is not None:
                        command()
                self.pausedflag = False

    def checkkey(self):
        """
        Check if self.key has been set and execute command accordingly.
        In order to add / modify key bindings, update the dict self.bindings with no-argument methods.

        key -- a char

        """
        try:
            if self.vmanager.imqueue is not None:
                key = self.key
                if type(key) is not str:
                    key = chr(key)
            else:
                key = chr(cv2.waitKey(50))  # for when running on main thread

            command = self.bindings[key]
            if command is not None:
                print("executing command '{0}'".format(key))
                command()
            else:
                print("no command for '{0}'".format(key))
        except (TypeError, KeyError, ValueError):
            pass  # not interested in non-char keys ATM
        self.key = None

    def undo(self):
        self.undoflag = True

    def _show(self, img, name="VidProcessor", latency=True, thread=False):
        """
        Offer the image to the main thread for display.

        """
        if latency:
            draw_str(img, (40, 20), "latency:  %.1f ms" % (self.latency * 1000))
        if thread:
            draw_str(img, (40, 40), "thread : " + current_thread().getName())
        if self.pausedflag:
            for img in self.own_images.values():
                draw_str(img, (img.shape[0]/2-30, img.shape[1] / 2), "PAUSED")
        try:
            if self.vmanager.imqueue is not None:
                self.vmanager.imqueue.put_nowait((name, img, self))
            else:
                show(img, name=name)  # assume we are on main thread
            self.own_images[name] = img
        except Full:
            print("Image queue full, not showing {0}".format(hex(id(img))))

    def _destroy_windows(self):
        """
        Multi-threaded env: ask for the windows created by this VidProcessor to be destroyed.
        Single-threaded env: destroy the windows created by this VidProcessor.

        """
        for name in self.own_images.keys():
            if self.vmanager.imqueue is not None:
                # caveat: wait until a slot is available to ensure destruction
                self.vmanager.imqueue.put((name, None, None))
            else:
                cv2.destroyWindow(name)
        self.own_images.clear()
