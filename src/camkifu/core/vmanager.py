from time import sleep
from threading import Thread, current_thread, Lock
import cv2
from os.path import isfile

from camkifu.config.cvconf import bfinders, sfinders, unsynced
from camkifu.core.video import VisionThread


__author__ = 'Arnaud Peloquin'


class VManagerBase(Thread):
    """
    Abstract vision manager, responsible for creating and coordinating all video detection processes.

    """

    def __init__(self, controller, imqueue=None):
        Thread.__init__(self, name="Vision")
        self.controller = controller
        self.controller.corrected = self.corrected
        self.imqueue = imqueue

        self.capt = None  # initialized in run() with video argument
        self.current_video = None

        self.board_finder = None
        self.stones_finder = None
        self.full_speed = False

    def init_capt(self):
        """
        Initialize the video capture with the source defined in the controller.

        """
        if self.capt is not None:
            self.capt.release()
        #noinspection PyArgumentList
        self.capt = cv2.VideoCapture(self.controller.video)
        if isfile(self.controller.video):
            self.capt = FileCaptureWrapper(self.capt)
            self.full_speed = True
        else:
            self.full_speed = False
        self.current_video = self.controller.video

        # set the beginning of video files. is ignored by live camera
        self.capt.set(cv2.CAP_PROP_POS_AVI_RATIO, self.controller.bounds[0])

    def run(self):
        raise NotImplementedError("Abstract method meant to be extended")

    def corrected(self, err_move, exp_move):
        """
        Inform that a correction has been made by the user on the goban.
        There is no need to perform any direct action, but this information can be used to tune detection.
        It is of high interest to prevent deleted stones to be re-suggested as soon as they are removed.

        Both moves below are expected to have their correct number set.
        err_move -- the faulty move (has been removed). can be None if the correction is an "add".
        exp_move -- the correct move (has been added). can be None if the correction is a "delete".

        """
        if self.stones_finder is not None:
            self.stones_finder.corrected(err_move, exp_move)

    def stop_processing(self):
        """
        Request this video manager to terminate all its sub-processes. The thread itself will not die,
        so that processing can be restarted later on.

        """
        raise NotImplementedError("Abstract method meant to be extended")

    def confirm_stop(self, process):
        """ A sub-process of this manager that terminates is supposed to pass itself here. """
        pass


class VManager(VManagerBase):
    """
    Multi-threaded vision manager.

    Its fields, notably the board and stones finders, must be regarded as (recursively) volatile.
    Concurrency issues are to be expected in this area.

    """

    def __init__(self, controller, imqueue=None):
        super(VManager, self).__init__(controller, imqueue=imqueue)
        self.daemon = True
        self.controller._pause = self._pause
        self.controller._on = self._on
        self.controller._off = self._off
        self.processes = []  # video processors currently running
        self.restart = False

    def _register_processes(self):
        # register "board finders" and "stones finders" with the controller,
        # together with callbacks to start them up.
        for bf_class in bfinders:
            self.controller.pipe("register_bf", (bf_class, self.set_bf))
        for sf_class in sfinders:
            self.controller.pipe("register_sf", (sf_class, self.set_sf))

    def run(self):
        """
        Run the main loop of VManager: provide to the controller all the finders listed in the config, and
        listen to video input changes.

        This main loop does not exit by itself. An exit can only occurs under the 'daemon thread' scheme.

        """
        if self.current_video != self.controller.video:
            self.init_capt()
        if not self.restart:  # only at startup
            self._register_processes()

        # start/restart processes
        self.set_bf(type(self.board_finder) if self.board_finder is not None else bfinders[0])
        self.set_sf(type(self.stones_finder) if self.stones_finder is not None else sfinders[0])

        # main loop, just watch for video input changes.
        self.restart = False
        while not self.restart:
            if self.current_video != self.controller.video:
                # global restart on the new video input
                self.stop_processing()
                self.restart = True
                break
            sleep(0.3)

        # do not allow VManager thread to terminate
        print("Vision processing restarting.")
        self.run()

    def is_processing(self):
        return len(self.processes).__bool__()  # make the return type clear

    def _on(self):
        if not self.is_processing():
            self.restart = True  # will take effect on the Vision thread, 'run' method

    def _off(self):
        if self.is_processing():
            self.stop_processing()
        self.controller.pipe("select_bf", [None])
        self.controller.pipe("select_sf", [None])

    def stop_processing(self):
        message = "Requesting "
        for proc in self.processes:
            proc.interrupt()
            message += proc.name + ", "
        message = message[0:len(message)-2]
        message += " interruption."
        # release a potential FileCaptureWrapper that may keep threads sleeping
        try:
            self.capt.unsync_threads(True)
        except AttributeError:
            pass
        print(message)

    def confirm_stop(self, process):
        self.processes.remove(process)
        print("{0} terminated.".format(process.__class__.__name__))
        # update a potential FileCaptureWrapper to wait for the right number of threads
        try:
            self.capt.nb_threads = len(self.processes)
        except AttributeError:
            pass

    def _spawn(self, process):
        """
        Start the provided process and append it to the list of active processes.

        """
        vt = VisionThread(process)
        process.full_speed = self.full_speed
        self.processes.append(vt)
        # reset a potential FileCaptureWrapper to normal behavior so that images flow again
        try:
            self.capt.unsync_threads(False)
            self.capt.nb_threads = len(self.processes)
        except AttributeError:
            pass
        print("{0} starting.".format(process.__class__.__name__))
        vt.start()

    def _pause(self, boolean):
        """
        Pause all sub-processes.

        """
        for process in self.processes:
            process.pause(boolean)

    def set_bf(self, bf_class):
        """
        Spawn a new instance of the provided board finder class, and terminate the previous board finder.

        """
        # delete previous instance if any
        if self.board_finder is not None:
            self.board_finder.interrupt()
            while self.board_finder in self.processes:
                sleep(0.1)
            self.board_finder = None  # may help prevent misuse
        # instantiate and start new instance
        self.board_finder = bf_class(self)
        self._spawn(self.board_finder)
        self.controller.pipe("select_bf", [self.board_finder.label])

    def set_sf(self, sf_class):
        """
        Terminate the current stone finder, and spawn a new instance of the provided stone finder class.

        """
        # delete previous instance if any
        if self.stones_finder is not None:
            self.stones_finder.interrupt()
            while self.stones_finder in self.processes:
                sleep(0.1)
            del self.stones_finder  # may help prevent misuse
        # instantiate and start new instance
        self.stones_finder = sf_class(self)
        self._spawn(self.stones_finder)
        self.controller.pipe("select_sf", [self.stones_finder.label])


class FileCaptureWrapper():
    """
    Wrapper to synchronize image consumption from a file : all "nb_threads" threads must have received the
    current image before the next image is consumed from VideoCapture.

    Delegates all _getattr()__ calls to the provided capture object, except for read() on which the
    synchronization is intended.

    -- capture : the object from which actual read() calls will be consumed.
    -- nb_threads : the number of threads to synchronize.

    """

    def __init__(self, capture, nb_threads=2):
        self.capture = capture
        self.nb_threads = nb_threads
        self.buffer = None   # the current result from videocapture.read()
        self.served = set()  # the threads that have received the currently buffered image
        self.lock_init = Lock()     # synchronization on videocapture.read() calls the first time
        self.lock_consume = Lock()  # synchronization on videocapture.read() calls
        self.sleep_time = 0.05  # 50 ms
        self.unsync = False     # True means that no thread should be kept sleeping, and the reading is stopped

    def __getattr__(self, item):
        """
        Hijack the "read()" method, delegate all others.

        """
        if item == "read":
            return self.read_sync
        else:
            return self.capture.__getattribute__(item)

    def read_sync(self):
        """
        Implements a custom read() method that provides the same image to all threads.
        Wait until all threads have been served before consuming the next frame.

        """
        self.init_buffer()
        thread = current_thread()
        while not self.unsync and thread in self.served:
            sleep(self.sleep_time)
        self.served.add(thread)
        self.consume()
        return self.buffer

    def init_buffer(self):
        with self.lock_init:
            if self.buffer is None:
                self.buffer = self.capture.read()

    def consume(self):
        """
        If unsync has been requested, set buffer to a None result.
        Else if all threads have been served, consume next image from videocapture (update buffer).
        Else do nothing.

        """
        with self.lock_consume:
            if self.unsync:
                self.buffer = False, unsynced
                self.served.clear()
            elif self.nb_threads <= len(self.served):
                assert self.nb_threads == len(self.served)  # todo remove after a few days testing
                self.buffer = self.capture.read()
                self.served.clear()

    def unsync_threads(self, unsync):
        """
        Request the release of all threads waiting in self.read_sync()

        -- unsync : True if all threads should be released, False to resume normal behavior.

        """
        self.unsync = unsync



