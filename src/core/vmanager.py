from time import sleep
import cv2
from threading import Thread
from board.board1 import BoardFinderManual
from board.board2 import BoardFinderAuto
from core.calib import Rectifier
from stone.stones1 import BackgroundSub
from stone.stones2 import NeighbourComp
from stone.stones4 import StoneCont

__author__ = 'Kohistan'


class VManagerBase(Thread):
    """
    Abstract vision manager, responsible for creating and coordinating all detection processes.

    """

    def __init__(self, controller, imqueue=None, video=0):
        Thread.__init__(self, name="Vision")
        self.controller = controller
        self.imqueue = imqueue

        self.video = video
        self.cam = None  # initialized in run() with video argument
        self.board_finder = None
        self.stones_finder = None

    def run(self):
        if self.cam is None:
            #noinspection PyArgumentList
            self.cam = cv2.VideoCapture(self.video)

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

    def __init__(self, controller, imqueue=None, video=0):
        super(VManager, self).__init__(controller, imqueue=imqueue, video=video)
        self.daemon = True
        self.controller._pause = self._pause
        self.processes = []  # video processors currently running

    def run(self):
        super(VManager, self).run()
        rectifier = Rectifier(self)

        # self.board_finder = BoardFinderManual(self, rectifier)
        self.board_finder = BoardFinderAuto(self, rectifier)
        self._spawn(self.board_finder)

        # self.stones_finder = BackgroundSub(self, rectifier)
        self.stones_finder = NeighbourComp(self, rectifier)
        self._spawn(self.stones_finder)

        running = 1
        while running:
            if self.stones_finder.undoflag:
                self.stones_finder.undoflag = False
                self.board_finder.perform_undo()
                if self.board_finder not in self.processes:
                    self._spawn(self.board_finder)
            running = len(self.processes)
            sleep(0.3)
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

    def _spawn(self, process):
        vt = VisionThread(process)
        self.processes.append(vt)
        vt.start()

    def _pause(self, boolean):
        for process in self.processes:
            process.pause(boolean)


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






















