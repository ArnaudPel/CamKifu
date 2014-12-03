from collections import defaultdict
from queue import Full
from threading import current_thread, Thread
from time import sleep, time

import cv2
from numpy import ndarray

from camkifu.config.cvconf import unsynced, frame_period
from camkifu.core.imgutil import show, draw_str, destroy_win


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
        self.last_read = 0.0  # the instant of the last successful frame read (before the processing of that frame).

        self._interruptflag = False
        self.pausedflag = False
        self.next_flag = False  # see self.next()

        self.bindings = {'p': self.pause, 'q': self.interrupt}
        self.key = None

        # Tkinter and openCV must cohabit on the main thread to display things.
        # If this main thread is too crammed with openCV showing images, bad things start to happen
        # The variables below help relieving the main thread by not showing all images requested.
        self.last_shown = 0  # the last time an image has been shown for this VidProcessor
        self.ignored_show = 0  # the number of images ignored since last_shown

        # metadata to print on the next image.
        # the structure is a dict because some images may not be shown, so data may have to be aggregated / overwritten
        # usage : for k, v in self.metadata.items(): k.format(v)  --  one key per row
        self.metadata = defaultdict(list)
        self.total_f_processed = 0  # total number of frames processed since init.

    def execute(self):
        """
        Execute the video processing loop. Run until self.interrupt() is called, or if the
        last frame has been reached when reading a file.

        In order to save some CPU, the loop sleeps between two iterations if they are processed
        faster than self.frame_period. Set this value to 0 to run flat out.

        """
        self._interruptflag = False
        while not self._interrupt_mainloop():
            self._checkpause()
            # check that the minimal time period between two iterations is respected
            frequency_condition = self.full_speed or (self.frame_period < time() - self.last_read)
            if self.ready_to_read() and frequency_condition:
                ret, frame = self.vmanager.read(self)
                if ret:
                    self.last_read = time()
                    self._doframe(frame)
                    self.total_f_processed += 1
                    self.checkkey()
                else:
                    if frame != unsynced:
                        print("Could not read camera for {0}.".format(str(type(self))))
                        sleep(5)
            else:
                sleep(self.frame_period / 10)  # precision doesn't really matter here
        self._destroy_windows()
        self.vmanager.confirm_stop(self)

    def _interrupt_mainloop(self) -> bool:
        """
        The condition evaluated by the main loop of self.execute(), True indicates interruption.
        Extension point.

        """
        return self._interruptflag or \
                    (self.vmanager.controller.bounds[1] <= self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO))

    def ready_to_read(self) -> bool:
        """
        Indicate if self is ready to consume frames from the VideoCapture. Extension point that can be used
        if a VidProcessor is waiting for something and will ignore frames passed down to it via self._doframe().

        This indication is crucial for file read synchronization : other threads will not be kept waiting on self if
        it is not ready to read (see class CaptureReader).

        """
        return not self._interruptflag

    def _doframe(self, frame: ndarray):
        """
        Image-processing algorithm may be implemented under that extension point.

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def interrupt(self):
        """
        Stop the video processing loop.

        """
        self._interruptflag = True

    def pause(self, dopause: bool=None):
        """
        Pause the video processing loop if "dopause" is True, resume loop if "dopause" is false.
        If "dopause" is not provided, toggle "paused" state.

        """
        if dopause is not None:
            # set provided value
            self.pausedflag = dopause
        else:
            # no value provided, interpret as a toggle
            self.pausedflag = not self.pausedflag

    def next(self):
        """
        Indicate that one frame may be allowed to be read if self is in "paused" state. Has no effect if self is
        not paused, since in this case frames are supposed to be flowing already.

        """
        self.next_flag = True

    def _checkpause(self):
        """
        Multi-threaded env: will sleep thread as long as self.pausedflag is True.
        Single-threaded env: will keep calling cv.waitKey until a valid command is pressed.

        """
        if self.vmanager.imqueue is not None:  # supposedly a multi-threaded env
            while self.pausedflag and not self.next_flag:
                sleep(0.1)
            self.next_flag = False
        else:  # supposedly in single-threaded dev mode
            if self.pausedflag:
                key = None
                while True:
                    # repeating the pause key resumes processing.
                    # any key in the bindings resumes processing (and the command is executed).
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

    def _draw_metadata(self, img, latency, thread):
        """
        Print info strings on the image.

        """
        # step 1 : draw default metadata, from the top of image
        try:
            percent_progress = int(100 * self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO))
            x_offset = 40
            line_spacing = 20
            draw_str(img, "video progress {0} %".format(percent_progress), x_offset, line_spacing)
            draw_str(img, "images not shown:  %d" % self.ignored_show, x_offset, 2 * line_spacing)
            if latency:
                draw_str(img, "latency:  %.1f ms" % ((time() - self.last_read) * 1000), x_offset, 3 * line_spacing)
            if thread:
                draw_str(img, "thread : " + current_thread().getName(), x_offset, 4 * line_spacing)
                # step 2 : draw custom metadata, from the bottom of image
            i = 0
            for k, v in self.metadata.items():
                draw_str(img, k.format(v), x_offset, img.shape[0] - (i + 1) * line_spacing)
                i += 1
            self.metadata.clear()
            if self.pausedflag:
                for img in self.own_images.values():
                    draw_str(img, "PAUSED", int(img.shape[0] / 2 - 30), int(img.shape[1] / 2))
        except Exception as exc:
            print("VidProcessor._draw_metadata(): {}".format(exc))

    def _show(self, img, name=None, latency=True, thread=False, loc=None, max_frequ=2):
        """
        Offer the image to the main thread for display.

        -- name : None lets the default name to be used (as per self._window_name()). One different window per name.
        -- thread : print current thread name on the image
        -- loc : the location of the window on the screen
        -- max_frequ : the max number of images displayed per second, when running with GUI (otherwise ignored).

        """
        if name is None:
            name = self._window_name()
        if self.vmanager.imqueue is not None:
            if 1 / max_frequ < time() - self.last_shown:  # todo use a dict to allow one VProc to show multiple windows
                self._draw_metadata(img, latency, thread)
                try:
                    self.vmanager.imqueue.put_nowait((name, img, self, loc))
                    self.own_images[name] = img
                    self.ignored_show = 0
                    self.last_shown = time()
                except Full:
                    self.ignored_show += 1
                    print("Image queue full, not showing {0}".format(hex(id(img))))
            else:
                self.ignored_show += 1
        else:
            self._draw_metadata(img, latency, thread)
            show(img, name=name, loc=loc)  # assume we are on main thread
            self.own_images[name] = img

    def _window_name(self) -> str:
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
                destroy_win(name)
        self.own_images.clear()


class VisionThread(Thread):
    """
    Wrapper for VidProcessor, to run it in a daemon thread.

    """
    def __init__(self, processor):
        super().__init__(name=processor.__class__.__name__)
        self.daemon = True
        self.processor = processor

        # delegate
        self.run = processor.execute

    def __getattr__(self, item):
        """
        Delegate all attributes to processor if not found in VisionThread.

        """
        # todo is this clean or should it be replaced with multiple inheritance ? + read doc again about __getattr__
        if item != "processor":  # avoid infinite looping if 'processor' attribute doesn't exist
            return self.processor.__getattribute__(item)

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