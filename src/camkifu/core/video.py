from queue import Full
from threading import current_thread, Thread
from time import sleep, time

import cv2

from camkifu.config.cvconf import unsynced, frame_period
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

        self.frame_period = frame_period  # shortest time between two processings: put thread to sleep if it's too early
        self.full_speed = False  # whether to respect the frame period or not
        self.last_read = 0.0  # gives the instant of the last image processing start (in seconds).

        self.undoflag = False  # todo move that down to ManualBoardFinder ?

        self._interruptflag = False
        self.pausedflag = False

        self.bindings = {'p': self.pause, 'q': self.interrupt, 'z': self.undo}
        self.key = None
        self.latency = 0.0

        self.metadata = []  # data to print on the next image, on element per row

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
            # check if we respect the minimal time period between two iterations
            if self.full_speed or (self.frame_period < start - self.last_read):
                self.last_read = start
                # todo provide a before_read() extension point to enable for skipping unnecessary read
                ret, frame = self.vmanager.capt.read()
                if ret:
                    do_frame_start = time()
                    self._doframe(frame)
                    self.latency = time() - do_frame_start
                    self.checkkey()
                else:
                    if frame != unsynced:
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
                    sleep(0.1)
        else:  # supposedly in single-threaded dev mode
            if self.pausedflag:
                key = None
                while True:
                    # repeating the pause key resumes processing. other keys are executed as if nothing happened
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

    def draw_metadata(self, img, latency, thread):
        """
        Print info strings on the image.

        """
        # step 1 : draw default metadata, from the top of image
        percent_progress = int(100 * self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO))
        x_offset = 40
        line_spacing = 20
        draw_str(img, x_offset, line_spacing, "video progress {0} %".format(percent_progress))
        if latency:
            draw_str(img, x_offset, 2*line_spacing, "latency:  %.1f ms" % (self.latency * 1000))
        if thread:
            draw_str(img, x_offset, 3*line_spacing, "thread : " + current_thread().getName())
        # step 2 : draw custom metadata, from the bottom of image
        for i, line in enumerate(self.metadata):
            draw_str(img, x_offset, img.shape[0] - (i+1)*line_spacing, line)
        self.metadata.clear()
        if self.pausedflag:
            for img in self.own_images.values():
                draw_str(img, int(img.shape[0] / 2 - 30), int(img.shape[1] / 2), "PAUSED")

    def _show(self, img, name=None, latency=True, thread=False, loc=None):
        """
        Offer the image to the main thread for display.

        """
        self.draw_metadata(img, latency, thread)
        try:
            if name is None:
                name = self._window_name()
            if self.vmanager.imqueue is not None:
                self.vmanager.imqueue.put_nowait((name, img, self, loc))
            else:
                show(img, name=name, loc=loc)  # assume we are on main thread
            self.own_images[name] = img
        except Full:
            print("Image queue full, not showing {0}".format(hex(id(img))))

    def _window_name(self):
        """
        Provide the name of the default display window of this VidProcessor, if any.
        Meant to be extended.

        """
        return "VidProcessor"

    def _destroy_windows(self):
        """
        Multi-threaded env: ask for the windows created by this VidProcessor to be destroyed.
        Single-threaded env: destroy the windows created by this VidProcessor.

        """
        for name in self.own_images.keys():
            if self.vmanager.imqueue is not None:
                # caveat: wait until a slot is available to ensure destruction
                self.vmanager.imqueue.put((name, None, None, None))
            else:
                cv2.destroyWindow(name)
        self.own_images.clear()


class VisionThread(Thread):
    """
    Wrapper for VidProcessor, to run it in a daemon thread.

    """
    def __init__(self, processor):
        super().__init__(name=processor.__class__.__name__)
        self.daemon = True

        # delegate
        self.run = processor.execute
        self.interrupt = processor.interrupt
        self.pause = processor.pause

    def __eq__(self, other):
        """
        Implementation that can match either a VisionThread or a VidProcessor.

        """
        try:
            return (self.run == other.execute) and (self.interrupt == other.interrupt)
        except AttributeError:
            try:
                return (self.run == other.run) and (self.interrupt == other.interrupt)
            except AttributeError:
                return False

    def __hash__(self, *args, **kwargs):
        return self.name.__hash__(*args, **kwargs)