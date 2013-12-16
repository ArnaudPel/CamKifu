from threading import Thread
from time import sleep
from board.board1 import BoardFinderManual
from board.board2 import BoardFinderAuto
from core.calib import Rectifier
from core.vmanager import VManagerBase
from stone.stones1 import BackgroundSub
#from stone.stones4 import StoneCont
from stone.stones2 import NeighbourComp
from stone.stones4 import StoneCont
from stone.stones5 import StonesHough

__author__ = 'Kohistan'


class VManagerSeq(VManagerBase):
    """
    Single-threaded vision manager, meant to be used during development only.

    Notably because, as of today, opencv show() and waitkey() must be run on the main thread.

    """

    def __init__(self, controller=None, video=0):
        super(VManagerSeq, self).__init__(controller, video=video)
        self.current_proc = None

    def run(self):
        super(VManagerSeq, self).run()
        rectifier = Rectifier(self)

        # self.board_finder = BoardFinderManual(self, rectifier)
        self.board_finder = BoardFinderAuto(self, rectifier)

        # self.stones_finder = BackgroundSub(self, rectifier)
        self.stones_finder = NeighbourComp(self, rectifier)
        # self.stones_finder = StoneCont(self, rectifier)
        # self.stones_finder = StonesHough(self, rectifier)

        states = ("board detection", "stones detection")
        state = states[0]

        while True:

            if state == states[0]:
                self.current_proc = self.board_finder
                ProcessKiller(self.board_finder, lambda: self.board_finder.mtx is not None).start()
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


class ProcessKiller(Thread):

    def __init__(self, process, condition):
        Thread.__init__(self, name="Killer({0})".format(process.__class__.__name__))
        self.daemon = True
        self.process = process
        self.condition = condition

    def run(self):
        while True:
            if self.condition():
                self.process.interrupt()
            sleep(0.1)



















