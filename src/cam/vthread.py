from threading import Thread
import cv2
from cam.board import BoardFinder
from cam.calib import Rectifier
from cam.stones import StonesFinder
from gui.pipewarning import PipeWarning

__author__ = 'Kohistan'


class Vision(Thread):
    """
    Master thread for image processing.
    May spawn children in the future.

    """

    def __init__(self, observer, images):
        Thread.__init__(self, name="Vision")
        self.observer = observer
        self.cam = cv2.VideoCapture(0)
        self.imqueue = images
        self.current_proc = None

    def run(self):
        rectifier = Rectifier(self.cam)
        board_finder = BoardFinder(self.cam, rectifier, self.imqueue)

        states = ("board detection", "stones detection")
        state = states[0]

        while True:

            if state == states[0]:
                self.current_proc = board_finder
                board_finder.execute()
                if board_finder.mtx is not None:
                    stones_finder = StonesFinder(self.cam, rectifier, self.imqueue, board_finder.mtx,
                                                 board_finder.size)
                    stones_finder.observers.append(self.observer)
                    state = states[1]
                else:
                    break

            elif state == states[1]:
                self.current_proc = stones_finder
                stones_finder.execute()
                if stones_finder.undoflag:
                    board_finder.perform_undo()
                    state = states[0]
                    stones_finder.undoflag = False
                else:
                    break

    def request_exit(self):
        print "requesting {0} exit.".format(self.current_proc.__class__.__name__)
        self.current_proc.interrupt()


