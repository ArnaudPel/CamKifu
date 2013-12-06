from Queue import Full, Empty, Queue
from core.pipewarning import PipeWarning
from gui.controller import Controller, ControllerBase

__author__ = 'Kohistan'


class ControllerV(Controller):
    """
    Extension responsible for handling Vision threads inputs.

    """

    def __init__(self, kifu, display, user_input):
        super(ControllerV, self).__init__(kifu, user_input, display)
        self.queue = Queue(10)
        # commands from background that have to be executed on the GUI thread.
        self.input.bind("<<execute>>", self._execute)

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

    def _execute(self, event):
        """
        See self.api for the list of executables.

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


class ControllerVSeq(ControllerBase):
    """
    Controller with no GUI that is supposed to be run in a single-threaded environment.

    """

    def __init__(self, kifu):
        super(ControllerVSeq, self).__init__(kifu)

    def pipe(self, instruction, args):
        self.api[instruction](*args)
