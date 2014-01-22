from Queue import Full
from time import sleep, time

from cv import CV_CAP_PROP_POS_AVI_RATIO as POS_RATIO
import cv2

from core.imgutil import show, draw_str


__author__ = 'Kohistan'


class VidProcessor(object):
    """
    Class meant to be extended by implementations of video processing.
    By default the execute() method will keep running, interruption must be asked when needed.

    """

    def __init__(self, vmanager):
        """
        self.frame_period -- (float) the minimum number of seconds between 2 iterations of frame processing

        """
        self.vmanager = vmanager
        self.own_images = {}

        self.frame_period = 0.2
        self.last_read = 0.0

        self.undoflag = False
        self._interruptflag = False
        self.pausedflag = False

        self.bindings = {'p': self.pause, 'q': self.interrupt, 'z': self.undo}
        self.key = None
        self.latency = 0.0

    def execute(self):
        end = self.vmanager.controller.bounds[1]
        self._interruptflag = False
        while not self._interruptflag and (self.vmanager.capt.get(POS_RATIO) < end):
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
                    print "Could not read camera for {0}.".format(str(type(self)))
                    self.latency = 0.0
                    sleep(5)
            else:
                sleep(self.frame_period / 10)  # precision doesn't really matter here
        self._destroy_windows()
        self.vmanager.confirm_exit(self)

    def checkkey(self):
        """
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
                print "executing command '{0}'".format(key)
                command()
            else:
                print "no command for '{0}'".format(key)
        except (TypeError, KeyError, ValueError):
            pass  # not interested in non-char keys ATM
        self.key = None

    def interrupt(self):
        self._interruptflag = True

    def pause(self, boolean=None):
        if boolean is not None:
            # set provided value
            self.pausedflag = boolean
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
                    for name, img in self.own_images.iteritems():
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

    def undo(self):
        self.undoflag = True

    def _show(self, img, name="VidProcessor", latency=True):
        """
        Offer the image to the main thread for display.

        """
        if latency:
            draw_str(img, (40, 20), "latency:  %.1f ms" % (self.latency * 1000))
        if self.pausedflag:
            for img in self.own_images.itervalues():
                draw_str(img, (img.shape[0]/2-30, img.shape[1] / 2), "PAUSED")
        try:
            if self.vmanager.imqueue is not None:
                self.vmanager.imqueue.put_nowait((name, img, self))
            else:
                show(img, name=name)  # assume we are on main thread
            self.own_images[name] = img
        except Full:
            print "Image queue full, not showing {0}".format(hex(id(img)))

    def _destroy_windows(self):
        for name in self.own_images.iterkeys():
            if self.vmanager.imqueue is not None:
                # caveat: wait until a slot is available to ensure destruction
                self.vmanager.imqueue.put((name, None, None))
            else:
                cv2.destroyWindow(name)
        self.own_images.clear()

    def _doframe(self, frame):
        raise NotImplementedError("Abstract method meant to be extended")