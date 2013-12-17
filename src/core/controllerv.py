from Queue import Full, Empty, Queue
from core.warnings import PipeWarning
from gui.controller import Controller, ControllerBase

__author__ = 'Kohistan'


class ControllerV(Controller):
    """
    Extension responsible for handling Vision threads inputs.

    """

    def __init__(self, user_input, display, kifufile=None):
        super(ControllerV, self).__init__(user_input, display, kifufile)
        self.queue = Queue(10)

        self.paused = Switch()  # alternate between paused and running state
        self.input.commands["pause"] = lambda: self._pause(self.paused.toggle())

        # commands from background that have to be executed on the GUI thread.
        self.input.bind("<<execute>>", self._cmd)

        self.api["bfinder"] = self.add_bfinder
        self.api["sfinder"] = self.add_sfinder

    def pipe(self, instruction, args):
        if self.input.closed:
            raise PipeWarning("Target User Interface has been closed.")
        if instruction == "event":
            # virtual event, comes from self.input itself, neither keyin nor mousein
            self.input.event_generate(args)
        else:
            try:
                self.queue.put_nowait((instruction, args))
            except Full:
                print "Goban instruction queue full, ignoring {0}".format(instruction)
            self.input.event_generate("<<execute>>")

    def _cmd(self, event):
        """
        See self.api for the list of executable commands.

        """
        try:
            while True:
                instruction, args = self.queue.get_nowait()
                try:
                    self.api[instruction](*args)
                except KeyError:
                    pass  # instruction not implemented here
        except Empty:
            pass

    def _pause(self, boolean):
        """
        To be set by Vision processes that would agree to pause on user demand.
        Pause if boolean is True, else resume.

        """
        pass

    def add_bfinder(self, label, callback, select=False):
        self.display.add_bf(label, callback, select=select)

    def add_sfinder(self, label, callback, select=False):
        self.display.add_sf(label, callback, select=select)

    def _open(self):
        self._pause(True)
        super(ControllerV, self)._open()
        self._pause(False)

    def _save(self):
        self._pause(True)
        super(ControllerV, self)._save()
        self._pause(False)

    def __setattr__(self, name, value):

        # special case for the current move number,
        if name == "current_mn" and value is not None:
            try:
                if value < self.kifu.last_move().number:
                    self._pause(True)
                else:
                    self._pause(self.paused.get())
            except AttributeError:
                pass  # most likely last_move() has returned null

        return super(ControllerV, self).__setattr__(name, value)


class ControllerVSeq(ControllerBase):
    """
    Controller with no GUI that is supposed to be run in a single-threaded environment.

    """

    def __init__(self, kifufile=None):
        super(ControllerVSeq, self).__init__(kifufile=kifufile)

    def pipe(self, instruction, args):
        self.api[instruction](*args)


class Switch(object):
    def __init__(self, on=False):
        self._on = on

    def toggle(self):
        """
        Return the current state and negate it.

        """
        self._on = not self._on
        return self._on

    def get(self):
        return self._on