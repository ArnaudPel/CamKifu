from time import sleep
import cv2
from threading import Thread
from cv2.cv import CV_CAP_PROP_POS_AVI_RATIO
from board.board1 import BoardFinderManual
from board.board2 import BoardFinderAuto
from core.calib import Rectifier
from stone.stones1 import BackgroundSub
from stone.stones2 import NeighbourComp
from stone.stones4 import StoneCont
from stone.stonesbase import DummyFinder

__author__ = 'Kohistan'


class VManagerBase(Thread):
    """
    Abstract vision manager, responsible for creating and coordinating all video detection processes.

    """

    def __init__(self, controller, imqueue=None):
        Thread.__init__(self, name="Vision")
        self.controller = controller
        self.imqueue = imqueue

        self.capt = None  # initialized in run() with video argument
        self.current_video = None

        self.board_finder = None
        self.stones_finder = None

    def init_capt(self):
        if self.capt is not None:
            self.capt.release()
        #noinspection PyArgumentList
        self.capt = cv2.VideoCapture(self.controller.video)
        self.current_video = self.controller.video

        # set the beginning of video files. is ignored by live camera
        self.capt.set(CV_CAP_PROP_POS_AVI_RATIO, self.controller.bounds[0])

    def request_exit(self):
        raise NotImplementedError("Abstract method meant to be extended")

    def confirm_exit(self, process):
        """ A process that terminates is supposed to pass itself here. """
        pass


class VManager(VManagerBase):
    """
    Multi-threaded vision manager.

    Its fields, notably the board and stones finders, must be regarded as (recursively) volatile.
    Concurrency issues are to be expected.

    """

    def __init__(self, controller, imqueue=None):
        super(VManager, self).__init__(controller, imqueue=imqueue)
        self.daemon = True
        self.controller._pause = self._pause
        self.processes = []  # video processors currently running
        self.restart = False

    def run(self):
        self.init_capt()
        rect = Rectifier(self)

        # register "board finders" and "stones finders" with the controller.
        # it's up to it to start them via the provided callbacks.
        self.controller.pipe("bfinder", ("Automatic", lambda: self.set_bf(BoardFinderAuto(self, rect))))
        self.controller.pipe("bfinder", ("Manual", lambda: self.set_bf(BoardFinderManual(self, rect)), True))

        dummy_sf = DummyFinder(self, rect, ["W[H8]", "B[J8]", "W[K12]", "B[F12]", "W[F11]", "B[H10]",
                                            "W[J14]", "B[J12]", "W[J11]", "B[J13]", "W[K13]"])
        self.controller.pipe("sfinder", ("Test SF", lambda: self.set_sf(dummy_sf)))
        self.controller.pipe("sfinder", ("Bg Sub", lambda: self.set_sf(BackgroundSub(self, rect)), True))
        self.controller.pipe("sfinder", ("Neigh Comp", lambda: self.set_sf(NeighbourComp(self, rect))))

        # todo remove that block and keep the bf_manual alive instead ?
        running = 1
        while running:
            # todo rethink this undo concept maybe
            if self.stones_finder.undoflag:
                self.stones_finder.undoflag = False
                self.board_finder.perform_undo()
                if self.board_finder not in self.processes:
                    self._spawn(self.board_finder)
            running = len(self.processes)

            # watch for video input changes.
            if self.current_video != self.controller.video:
                # global restart to avoid fatal "PyEval_RestoreThread: NULL tstate"
                self.restart = True
                self.request_exit()
            sleep(0.3)

        if self.restart:
            self.restart = False
            print "Vision processing restarting."
            self.run()
        else:
            print "Vision processing terminated."

    def request_exit(self):
        message = "Requesting "
        for proc in self.processes:
            proc.interrupt()
            message += proc.name + ", "
        message = message[0:len(message)-2]
        message += " interruption."
        print message

    def confirm_exit(self, process):
        self.processes.remove(process)
        print "{0} terminated.".format(process.__class__.__name__)
        if process is self.board_finder and process.mtx is None:
            # board finder exits without providing board location. that's a show stopper
            self.request_exit()

    def _spawn(self, process):
        vt = VisionThread(process)
        self.processes.append(vt)
        print "{0} starting.".format(process.__class__.__name__)
        vt.start()

    def _pause(self, boolean):
        for process in self.processes:
            process.pause(boolean)

    def set_bf(self, board_finder):
        # always have at least one bf running to keep sf alive
        tostop = self.board_finder

        self.board_finder = board_finder
        self._spawn(self.board_finder)

        if tostop is not None:
            tostop.interrupt()
            while tostop in self.processes:
                sleep(0.1)
            del tostop  # may help prevent misuse

    def set_sf(self, stones_finder):
        if self.stones_finder is not None:
            self.stones_finder.interrupt()
            while self.stones_finder in self.processes:
                sleep(0.1)
            del self.stones_finder  # may help prevent misuse
        self.stones_finder = stones_finder
        self._spawn(self.stones_finder)


class VisionThread(Thread):
    """
    Wrapper for VidProcessor, to run it in a daemon thread.

    """
    def __init__(self, processor):
        super(VisionThread, self).__init__(name=processor.__class__.__name__)
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




















