from time import sleep
from threading import Thread

import cv2

from camkifu.config.cvconf import bfinders, sfinders


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

    def init_capt(self):
        """
        Initialize the video capture with the source defined in the controller.

        """
        if self.capt is not None:
            self.capt.release()
        #noinspection PyArgumentList
        self.capt = cv2.VideoCapture(self.controller.video)
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
        self.controller.pipe("select_bf", [self.board_finder.label])
        self.set_sf(type(self.stones_finder) if self.stones_finder is not None else sfinders[0])
        self.controller.pipe("select_sf", [self.stones_finder.label])

        # main loop, just watch for video input changes.
        self.restart = False
        while not self.restart:
            if self.current_video != self.controller.video:
                # global restart to avoid fatal "PyEval_RestoreThread: NULL tstate"
                self.stop_processing()
                self.restart = True
                break
            sleep(0.3)

        # do not allow thread to terminate
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
        print(message)

    def confirm_stop(self, process):
        self.processes.remove(process)
        print("{0} terminated.".format(process.__class__.__name__))

    def _spawn(self, process):
        """
        Start the provided process and append it to the list of active processes.

        """
        vt = VisionThread(process)
        self.processes.append(vt)
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
            del self.board_finder  # may help prevent misuse
        # instantiate and start new instance
        self.board_finder = bf_class(self)
        self._spawn(self.board_finder)

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