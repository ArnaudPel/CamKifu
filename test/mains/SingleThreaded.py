from threading import Thread
from time import sleep

from CkMain import get_argparser
from camkifu.board.bf_manual import BoardFinderManual
from camkifu.config.cvconf import bfinders, sfinders
from camkifu.core.vmanager import VManagerBase
from golib.gui.controller import ControllerBase


__author__ = 'Arnaud Peloquin'

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
        self.api = {
            "append": self.cvappend,
            "delete": self._delete,
            "bulk": self._bulk_append,

        }

    def pipe(self, instruction, args):
        """
        Execute instruction straight away (assumption of single-threaded environment).

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

    states = ("board detection", "stones detection", "stop")

    def __init__(self, controller=None):
        super(VManagerSeq, self).__init__(controller)
        self.state = VManagerSeq.states[0]
        self.current_proc = None
        self.bf_locked = False  # special flag for board finder manual which has to be kept running in some situations

    def init_bf(self):
        self.board_finder = bfinders[0](self)
        self.board_finder.bindings['o'] = self.unlock_bf
        self.board_finder.bindings['q'] = self.stop_processing

    def init_sf(self):
        self.stones_finder = sfinders[0](self)
        self.stones_finder.bindings['z'] = self.goto_detect
        self.stones_finder.bindings['q'] = self.stop_processing

    def goto_detect(self):
        print("requesting return to board detection state")
        self.state = VManagerSeq.states[0]
        # special for manual board finder : it must not be killed although it has got a board location, to allow for
        # user to correct as many points as needed. See self.init_bf() for the key binding giving the 'ok' signal.
        self.bf_locked = isinstance(self.board_finder, BoardFinderManual)
        self.stones_finder.interrupt()

    def unlock_bf(self):
        self.bf_locked = False

    def run(self):
        self.init_capt()
        self.init_bf()
        self.init_sf()
        while True:
            if self.state == VManagerSeq.states[0]:
                self.current_proc = self.board_finder
                stop_condition = lambda: not self.bf_locked and self.board_finder.mtx is not None
                ProcessKiller(self.board_finder, stop_condition).start()
                self.board_finder.execute()
                if self.state == VManagerSeq.states[0] and self.board_finder.mtx is not None:
                    self.state = VManagerSeq.states[1]
                else:
                    break
            elif self.state == VManagerSeq.states[1]:
                self.current_proc = self.stones_finder
                self.stones_finder.execute()
            else:
                break

    def stop_processing(self):
        print("requesting {0} exit.".format(self.current_proc.__class__.__name__))
        self.state = VManagerSeq.states[2]
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
