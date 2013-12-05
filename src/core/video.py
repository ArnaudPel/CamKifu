from Queue import Full
from threading import Thread
import cv2
import time
from core.imgutil import show, draw_str

__author__ = 'Kohistan'


class VidProcessor(object):
    """
    Class meant to be extended by implementations of video processing.
    By default the execute() method will keep running, interruption must be asked when needed.

    """

    def __init__(self, vmanager, rectifier=None):
        """
        self.frame_period -- (float) the minimum number of seconds between 2 iterations of frame processing

        """
        self.vmanager = vmanager
        self.rectifier = rectifier
        self.own_images = set()

        self.frame_period = 0.2
        self.last_frame = 0.0
        self.undoflag = False
        self._interruptflag = False

        self.bindings = {'p': self.pause, 'q': self.interrupt, 'z': self.undo}
        self.key = None
        self.latency = 0.0

    def execute(self):
        self._interruptflag = False
        while not self._interruptflag:
            start = time.time()
            if self.frame_period < start - self.last_frame:
                self.last_frame = start
                ret, frame = self.vmanager.cam.read()
                if ret:
                    # todo remove calibration if it's not actually helping.
                    # undistort seems to actually pollute board detection.
                    #if self.rectifier is not None:
                        #frame = self.rectifier.undistort(frame)
                    self._doframe(frame)
                    self.checkkey()
                    self.latency = time.time() - start
                else:
                    print "Could not read camera for {0}.".format(str(type(self)))
                    self.latency = 0.0
                    time.sleep(5)
            else:
                time.sleep(self.frame_period / 10)  # precision doesn't really matter here
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
            print key
            if command is not None:
                print "executing command '{0}'".format(key)
                command()
        except (TypeError, KeyError, ValueError):
            pass  # not interested in non-char keys ATM
        self.key = None

    def interrupt(self):
        self._interruptflag = True
        self._destroy_windows()

    def pause(self):
        # todo refactor for multi-threaded arch
        if self.vmanager.imqueue is None:
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

    def undo(self):
        self.undoflag = True

    def _show(self, img, name="VidProcessor", latency=True):
        """
        Offer the image to the main thread for display.

        """
        if latency:
            draw_str(img, (40, 20), "latency:  %.1f ms" % (self.latency * 1000))
        try:
            if self.vmanager.imqueue is not None:
                self.vmanager.imqueue.put_nowait((name, img, self))
            else:
                show(img, name=name)  # assume we are on main thread
            self.own_images.add(name)
        except Full:
            print "Image queue full, not showing {0}".format(hex(id(img)))

    def _destroy_windows(self):
        for name in self.own_images:
            if self.vmanager.imqueue is not None:
                # caveat: wait until a slot is available to ensure destruction
                self.vmanager.imqueue.put((name, None, None))
            else:
                cv2.destroyWindow(name)
        self.own_images.clear()

    def _doframe(self, frame):
        raise NotImplementedError("Abstract method meant to be extended")


class VidRecorder(VidProcessor):
    """
    Class to record a sample into a video file.

    """
    def __init__(self, camera, directory, filename, fps=5):
        """
        fps -- frames per second.
        filepath -- the location where to save the video

        """
        super(VidRecorder, self).__init__(camera)
        self.uri = "{0}/{1}.avi".format(directory, filename)
        self._fps = fps
        self._lastw = 0
        self._writer = None

    def _doframe(self, frame):
        # need to wait the first frame to pass down frameSize
        if self._writer is None:
            # not working on my mac. see http://stackoverflow.com/a/4885196/777285
            self._writer = cv2.VideoWriter(self.uri, cv2.cv.CV_FOURCC('M', 'P', 'G', '4'), self._fps, frame.shape[0:2], isColor=1)
        now = time.time()
        if 1.0 / self._fps < now - self._lastw:
            self._writer.write(frame)
            self._lastw = now


class KeyboardInput(Thread):
    def __init__(self, obs):
        Thread.__init__(self, name="KBIn")
        self.observer = obs
        self.interruptflag = False

    def run(self):
        prompt = "{0} listening to keyboard: \n".format(self.observer.__class__.__name__)
        try:
            while not self.interruptflag:
                self.observer.key = raw_input(prompt)
                time.sleep(0.25)
        except EOFError:
            print("KeyboardInput terminated.")

    def request_exit(self):
        self.interruptflag = True



























