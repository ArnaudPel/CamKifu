from time import sleep
import cv2
from threading import Thread
from board.board1 import BoardFinderManual
from board.board2 import BoardFinderAuto
from core.calib import Rectifier
from stone.stones1 import BackgroundSub
from stone.stones3 import BackgroundSub2

__author__ = 'Kohistan'


class VManager(Thread):
    """
    Master thread for image processing.

    Its fields, notably the board and stones finders, must be regarded as (recursively) volatile.
    Concurrency issues are to be expected.

    """

    def __init__(self, controller, images):
        Thread.__init__(self, name="Vision")

        #noinspection PyArgumentList
        self.cam = cv2.VideoCapture(0)
        self.controller = controller
        self.imqueue = images

        self.processes = []  # video processors currently running
        self.board_finder = None
        self.stones_finder = None

    def run(self):
        rectifier = Rectifier(self)

        #self.board_finder = BoardFinderManual(self, rectifier)
        self.board_finder = BoardFinderAuto(self, rectifier)
        self.spawn(self.board_finder)

        #self.stones_finder = NeighbourComp(self, rectifier)
        self.stones_finder = BackgroundSub(self, rectifier)
        self.spawn(self.stones_finder)

        running = 1
        while running:
            if self.stones_finder.undoflag:
                self.stones_finder.undoflag = False
                self.board_finder.perform_undo()
                if self.board_finder not in self.processes:
                    self.spawn(self.board_finder)
            running = len(self.processes)
            sleep(0.3)
        print "Vision processing terminated."

    def spawn(self, process):
        vt = VisionThread(process)
        self.processes.append(vt)
        vt.start()

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


class VisionThread(Thread):

    def __init__(self, processor):
        super(VisionThread, self).__init__(name=processor.__class__.__name__)

        # delegate
        self.run = processor.execute
        self.interrupt = processor.interrupt

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






















