import collections
import queue
import threading
import time
import traceback

import cv2
import numpy as np

from camkifu.config import cvconf
from camkifu.core import imgutil


__author__ = 'Arnaud Peloquin'


class VidProcessor(object):
    """ Abstract base class for periodic video processing.

    Periodically read frames from the vmanager, and pass them down to the abstract _doframe() method.
    The aim of this class is to provide basic functionalities around periodic frame reading, like pausing,
    interrupting, tuning the read frequency, or handling the display of images.

    By default the execute() method may keep running (eg. if the video input is live), interruption must be asked.

    Attributes:
        vmanager: VManager
            The manager handling this instance. It holds the video capture object as well as an image queue
            into which images to be displayed can be put. It would like to know when this instance
            terminates, and if something went wrong.
        bindings: dict
            Keyboard binding to methods, that will be used together with cv2.waitKey()
        key: a char
            The last key returned by cv2.waitKey()
        total_f_processed: int
            The total number of frames successfully passed down to _doframe() since the beginning .

        frame_period: float
            Shortest time (s) between two processing iterations: put thread to sleep if needs be.
        full_speed: bool
            Whether to respect the frame period or not.
        last_read: float
            The instant of the last successful frame read, before the processing of that frame.
        _interruptflag: bool
            True to indicate that the main execute() loop should be interrupted.
        pausedflag: bool
            True to indicate that the main execute() loop should be paused until further notice.
        next_flag: bool
            True to indicate that, if execute() is paused, on more iteration may be performed.

        own_images: dict
            The images that have been sent to display and not destroyed yet.
        last_shown: defaultdict
            Map image name to the last time it has been displayed.
        ignored_show: defaultdict
            Map image name to the number of times it has been ignored since last_shown.
        metadata: defaultdict
            Metadata to print on the next image. The structure is a dict because some images may not be shown, so
            data may have to be aggregated / overwritten.
            Usage : for k, v in metadata.items(): k.format(v)  # one key per row
    """

    def __init__(self, vmanager):
        """
        Args:
            vmanager: VManager
                The manager responsible for handling this vidprocessor instance. It holds the video capture object
                as well as an image queue into which images to be displayed can be put. It would like to know when
                this instance terminates, and if something went wrong.
        """
        self.vmanager = vmanager
        self.bindings = {'p': self.pause, 'q': self.interrupt, 'f': self.next}
        self.key = None
        self.total_f_processed = 0

        self.frame_period = cvconf.frame_period
        self.full_speed = False
        self.last_read = 0.0
        self._interruptflag = False
        self.pausedflag = False
        self.next_flag = False

        self.own_images = {}
        # Tkinter and openCV must currently cohabit on the main thread to display things.
        # If the main thread is too crammed with openCV showing images, bad things start to happen
        # The variables below help relieving the main thread by not showing all images requested.
        self.last_shown = collections.defaultdict(lambda: 0)
        self.ignored_show = collections.defaultdict(lambda: 0)
        self.metadata = collections.defaultdict(list)

    def execute(self):
        """ Execute the main video processing loop.

        Periodically read frames from vmanager and pass them down to the abstact _dofame(). Between each iteration,
        check for different instructions (interrupt, pause, read next, key pressed). Sleep between two iterations if
        they happen to complete faster than the required frame_period.
        Inform self.vmanager on termination.
        """
        try:
            self._interruptflag = False
            while not self._interrupt_mainloop():
                self._checkpause()
                # check that the minimal time period between two iterations is respected
                frequency_condition = self.full_speed or (self.frame_period < time.time() - self.last_read)
                if self.ready_to_read() and frequency_condition:
                    ret, frame = self.vmanager.read(self)
                    if ret:
                        self.last_read = time.time()
                        self._doframe(frame)
                        self.total_f_processed += 1
                        self.checkkey()
                    else:
                        if frame != cvconf.unsynced:
                            print("Could not read camera for {0}.".format(str(type(self))))
                            time.sleep(2)
                else:
                    time.sleep(self.frame_period / 10)  # precision doesn't really matter here
        except BaseException as exc:
            self.vmanager.error_raised(self, exc)
            traceback.print_exc()
        finally:
            self._destroy_windows()
            self.vmanager.confirm_stop(self)

    def _interrupt_mainloop(self) -> bool:
        """ Main loop interruption condition.

        Return: bool
            True (interrupt) if the flag has been set, or if the end of the video is reached, otherwise False.
        """
        if self.terminated_video():
            print("Video end reached. {} terminating main loop.".format(self.__class__.__name__))
            return True
        return self._interruptflag

    def terminated_video(self):
        """
        Return: bool
            True if the end of the video has been reached, False otherwise.
        """
        return self.vmanager.controller.bounds[1] <= self.vmanager.capt.get(cv2.CAP_PROP_POS_AVI_RATIO)

    def ready_to_read(self) -> bool:
        """ Indicate if self is ready to consume frames from the VideoCapture.

        Extension point that can be used if a VidProcessor is waiting for something and will ignore frames passed down
        to it via self._doframe(). Can spare useless frame consumption. This indication is crucial for file read
        synchronization : other threads will not be kept waiting on self if it doesn't ready any frame.

        Return: bool
        """
        return not self._interruptflag

    def _doframe(self, frame: np.ndarray):
        """ Image-processing algorithms may be implemented under that extension point.

        Args:
            frame: ndarray
                A raw image from input, that has to be analysed.
        """
        raise NotImplementedError("Abstract method meant to be extended")

    def interrupt(self):
        """ Request the interruption of the main video processing loop.
        """
        self._interruptflag = True

    def pause(self, dopause: bool=None):
        """ Toggle pause of the main video processing loop, or set it as per 'dopause' if it is provided.
        """
        if dopause is not None:
            self.pausedflag = dopause
        else:
            # no value provided, interpret as a toggle
            self.pausedflag = not self.pausedflag

    def next(self):
        """ Request one new frame read if main processing loop is in "paused" state. Has no effect otherwise.
        """
        self.next_flag = True

    def _checkpause(self):
        """ Check if the 'pause' flag has been set, and react accordingly.

        Multi-threaded env: will sleep thread as long as self.pausedflag is True.
        Single-threaded env: will keep calling cv.waitKey until a valid command is pressed.
        """
        if self.vmanager.imqueue is not None:  # supposedly a multi-threaded env
            while self.pausedflag and not self.next_flag:
                time.sleep(0.1)
            self.next_flag = False
        else:  # supposedly in single-threaded (dev) mode
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
        """ Check if self.key has been set, and react accordingly (call appropriate command).
        Supposed to be used in single threaded environment only.
        """
        try:
            if self.vmanager.imqueue is None:  # supposedly in single-threaded (dev) mode
                key = chr(cv2.waitKey(50))
                command = self.bindings[key]
                if command is not None:
                    print("executing command '{0}'".format(key))
                    command()
                else:
                    print("no command for '{0}'".format(key))
        except (TypeError, KeyError, ValueError):
            pass  # not interested in non-char keys ATM
        self.key = None

    def _draw_metadata(self, img: np.ndarray, name: str, latency: bool, thread: bool):
        """ Print info strings on the image based on the current instance data.

        Custom metadata is read from self.metadata, and printed as follows (one key per row):
        for k, v in metadata.items(): k.format(v)

        Args:
            img: ndarray
                The image on which to print the data.
            name: str
                The name of the image.
            latency: bool
                True if latency information should be printed.
            thread: bool
                True if current thread information should be printed.
        """
        # step 1 : draw default metadata, starting at the top of image
        try:
            # line 1
            frame_idx = int(round(self.vmanager.capt.get(cv2.CAP_PROP_POS_FRAMES)))
            total = int(round(self.vmanager.capt.get(cv2.CAP_PROP_FRAME_COUNT)))
            x_offset = 40
            line_spacing = 20
            s = "Frame {}/{} ({} not shown)".format(frame_idx, total, self.ignored_show[name])
            imgutil.draw_str(img, s, x_offset, line_spacing)
            # line 2 (optional)
            if latency:
                imgutil.draw_str(img, "latency: %.1f ms" % ((time.time() - self.last_read) * 1000), x_offset, 2 * line_spacing)
            # line 3 (optional)
            if thread:
                imgutil.draw_str(img, "thread: " + threading.current_thread().getName(), x_offset, 3 * line_spacing)

            # step 2 : draw custom metadata, starting at the bottom of image
            i = 0
            for k, v in self.metadata.items():
                imgutil.draw_str(img, k.format(v), x_offset, img.shape[0] - (i + 1) * line_spacing)
                i += 1
            self.metadata.clear()
            if self.pausedflag:
                for img in self.own_images.values():
                    imgutil.draw_str(img, "PAUSED", int(img.shape[0] / 2 - 30), int(img.shape[1] / 2))
        except Exception as exc:
            print("VidProcessor._draw_metadata(): {}".format(exc))

    def _show(self, img: np.ndarray, name: str=None, latency: bool=True, thread: bool=False,
              loc: (int, int)=None, max_frequ: float=2):
        """ Request an image to be shown / updated.

        Multi-threaded env: put 'img' into the image queue with respect to 'max_frequ'.
        Single-threaded env: call cv2.imshow() on the current thread immediately.

        Args
            img: ndarray
                The image to be shown.
            name: str
                The image name (=window name). Defaults to self._window_name(). If a window with that name already
                exists, it is updated.
            thread: bool
                Print current thread name on the image.
            loc: (int, int)
                The location of the window on the screen.
            max_frequ: float
                The max number of images displayed per second, when running with GUI (otherwise ignored). If this
                method is called too often with regards to this argument, the necessary number of images will be
                ignored.
        """
        if name is None:
            name = self._window_name()
        if self.vmanager.imqueue is not None:  # supposedly a multi-threaded env
            if 1 / max_frequ < time.time() - self.last_shown[name]:
                self._draw_metadata(img, name, latency, thread)
                try:
                    self.vmanager.imqueue.put_nowait((name, img, self, loc))
                    self.own_images[name] = img
                    self.ignored_show[name] = 0
                    self.last_shown[name] = time.time()
                except queue.Full:
                    self.ignored_show[name] += 1
                    print("Image queue full, not showing {0}".format(hex(id(img))))
            else:
                self.ignored_show[name] += 1
        else:  # supposedly in single-threaded (dev) mode
            self._draw_metadata(img, name, latency, thread)
            imgutil.show(img, name=name, loc=loc)  # assume we are on main thread
            self.own_images[name] = img

    def _window_name(self) -> str:
        """ Return the default window name for image display.
        """
        return "camkifu.core.video.VidProcessor"

    def _destroy_windows(self):
        """ Request the destruction of all the images shown by self.

        Multi-threaded env: put special tuple values in the image queue to request windows destruction.
        Single-threaded env: destroy the windows on this thread immediately.
        """
        for name in self.own_images.keys():
            if self.vmanager.imqueue is not None:
                # need to wait until a slot is available to ensure proper destruction.
                self.vmanager.imqueue.put((name, None, None, None))
            else:
                imgutil.destroy_win(name)
        self.own_images.clear()


class VisionThread(threading.Thread):
    """ Wrapper for VidProcessor, allowing it to run in a (daemon) thread.

    Attributes:
        daemon: bool
            Inherited from Thread, forced to True.
        processor: VidProcesssor
            The delegate object, notably responsible for the implementation of Thread.run().
    """
    def __init__(self, processor):
        super().__init__(name=processor.__class__.__name__)
        self.daemon = True
        self.processor = processor

        # delegate
        self.run = processor.execute

    def __getattr__(self, item):
        """ Delegate all __getattr__ to processor if not found in VisionThread.
        """
        if item != "processor":  # avoid infinite looping if 'processor' attribute doesn't exist
            return self.processor.__getattribute__(item)

    def __eq__(self, other):
        """ Allow a VisionThread to be equal to a VidProcessor, based on respective run()/execute() methods equality.
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