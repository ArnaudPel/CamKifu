from threading import Thread
from time import sleep
from Camkifu import get_argparser
from config.cvconf import bfinders, sfinders
from core.vmanager import VManagerBase
from gui.controller import ControllerBase

__author__ = 'Kohistan'

"""
Script that can be used to run all vision on the main thread. The counterpart is that no Tkinter
GUI can be used, as it has to monopolize the main thread.

This is mainly useful when in need to display something and pause in the middle of a vision algo,
especially to use waitKey().

"""


class ControllerVSeq(ControllerBase):
    """
    Controller with no GUI that is supposed to be run in a single-threaded environment.

    """

    def __init__(self, sgffile=None, video=0, bounds=(0, 1)):
        super(ControllerVSeq, self).__init__(sgffile=sgffile)
        self.video = video
        self.bounds = bounds
        self.api = {"append": self.cvappend}

    def pipe(self, instruction, args):
        """
        Execute command straight away (assumption of single-threaded environment).

        """
        self.api[instruction](*args)

    def cvappend(self, move):
        move.number = self.current_mn + 1
        self.rules.put(move)
        self._append(move)


class VManagerSeq(VManagerBase):
    """
    Single-threaded vision manager, meant to be used during development only (no GUI).
    Notably because, as of today, opencv show() and waitkey() must be run on the main thread.

    """

    def __init__(self, controller=None):
        super(VManagerSeq, self).__init__(controller)
        self.current_proc = None

    def run(self):
        self.init_capt()
        self.board_finder = bfinders[0](self)
        self.stones_finder = sfinders[0](self)

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


def main(video=0, sgf=None, bounds=(0, 1)):
    # run in dev mode, everything on the main thread
    vision = VManagerSeq(ControllerVSeq(sgffile=sgf, video=video, bounds=bounds))
    vision.run()

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, sgf=args.sgf, bounds=args.bounds)
