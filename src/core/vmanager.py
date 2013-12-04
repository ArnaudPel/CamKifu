import cv2
from threading import Thread
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

        self.current_proc = None  # video processor currently running
        self.board_finder = None
        self.stones_finder = None

    def run(self):
        rectifier = Rectifier(self)

        #board_finder = BoardFinderManual(self, rectifier)
        self.board_finder = BoardFinderAuto(self, rectifier)

        self.stones_finder = BackgroundSub(self, rectifier)
        #stones_finder = NeighbourComp(self, rectifier)

        states = ("board detection", "stones detection")
        state = states[0]

        while True:

            if state == states[0]:
                self.current_proc = self.board_finder
                self.board_finder.execute()
                if self.board_finder.mtx is not None:
                    state = states[1]
                else:
                    break

            elif state == states[1]:
                self.current_proc = self.stones_finder
                self.stones_finder.execute()
                if self.stones_finder.undoflag:
                    self.board_finder.perform_undo()
                    state = states[0]
                    self.stones_finder.undoflag = False
                else:
                    break

    def request_exit(self):
        print "requesting {0} exit.".format(self.current_proc.__class__.__name__)
        self.current_proc.interrupt()


