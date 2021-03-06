import threading
import time
from camkifu.board import bf_manual
import camkifu.core


class VManagerSeq(camkifu.core.VManagerBase):
    """
    Single-threaded vision manager, meant to be used during development only (no GUI).
    Notably because, as of opencv 3.0.0-beta, cv2.show() and cv2.waitkey() must be run on the main thread.

    """

    states = ("board detection", "stones detection", "stop")

    def __init__(self, controller=None, bf=None, sf=None):
        super().__init__(controller, bf=bf, sf=sf)
        self.state = VManagerSeq.states[0]
        self.current_proc = None
        self.bf_locked = False  # special flag for board finder manual which has to be kept running in some situations

    def init_bf(self):
        self.board_finder = self.bf_class(self)
        self.setup_finder(self.board_finder)
        self.board_finder.bindings['o'] = self.unlock_bf

    def init_sf(self):
        self.stones_finder = self.sf_class(self)
        self.setup_finder(self.stones_finder)
        self.stones_finder.bindings['z'] = self.goto_detect

    def setup_finder(self, finder):
        finder.bindings['q'] = self.stop_processing
        finder.full_speed = self.full_speed

    def goto_detect(self):
        print("requesting return to board detection state")
        self.state = VManagerSeq.states[0]
        # special for manual board finder : it must not be killed although it has got a board location, to allow for
        # user to correct as many points as needed. See self.init_bf() for the key binding giving the 'ok' signal.
        self.bf_locked = isinstance(self.board_finder, bf_manual.BoardFinderManual)
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
                if self.stones_finder.terminated_video():
                    break
            else:
                break

    def interrupt(self):
        """
        No difference between interruption an stopping processing for this single threaded VManager.

        """
        self.stop_processing()

    def stop_processing(self):
        """
        Stop current processor and exit "run" loop.

        """
        print("requesting {0} exit.".format(self.current_proc.__class__.__name__))
        self.state = VManagerSeq.states[2]
        self.current_proc.interrupt()


class ProcessKiller(threading.Thread):

    def __init__(self, process, condition):
        threading.Thread.__init__(self, name="Killer({0})".format(process.__class__.__name__))
        self.daemon = True
        self.process = process
        self.condition = condition

    def run(self):
        while True:
            if self.condition():
                self.process.interrupt()
                break
            time.sleep(0.1)
