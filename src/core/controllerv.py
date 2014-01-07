from Queue import Full, Empty, Queue
from core.warnings import PipeWarning
from gui.controller import Controller, ControllerBase

__author__ = 'Kohistan'

commands_size = 10


class ControllerV(Controller):
    """
    Extension adding handling of Vision threads to the default GUI controller.

    """

    def __init__(self, user_input, display, kifufile=None, video=0, bounds=(0, 1)):
        super(ControllerV, self).__init__(user_input, display, kifufile=kifufile)
        self.queue = Queue(commands_size)
        self.video = video
        self.bounds = bounds

        # commands from background that have to be executed on the GUI thread.
        self.input.bind("<<execute>>", self._cmd)

        try:
            self.input.commands["run"] = self._run
            self.input.commands["pause"] = lambda: self._pause(self.paused.true())
            self.input.commands["vidfile"] = self._openvideo
            self.input.commands["vidlive"] = self._openlive
        except AttributeError as ae:
            self.log("Some commands could not be bound to User Interface.")
            self.log(ae)

        self.api["bfinder"] = self.add_bfinder
        self.api["sfinder"] = self.add_sfinder

        if kifufile is not None:
            self._goto(722)  # get kifu ready to ramble

        self.paused = Pause()

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
                print "Controller instruction queue full, ignoring {0}".format(instruction)
            self.input.event_generate("<<execute>>")

    def _cmd(self, event):
        """
        Try to empty the piped commands queue.
        This method will not execute more than a fixed number of commands, in order to prevent
        infinite looping. Such infinite looping could occur if this method fails to keep up with
        other threads piping commands.
        Keeping flow smooth is paramount here, as this is likely to be run on the main (GUI) thread.

        See self.api for the list of executable commands.

        """
        try:
            count = 0
            while count < commands_size:
                count += 1
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

    def _run(self):
        """
        Ask vision processes to run / resume.

        """
        if self.at_last_move():
            self._pause(self.paused.false())
        else:
            self.log("Processing can't create variation in game. Please navigate to the last move.")

    def add_bfinder(self, label, callback, select=False):
        self.display.add_bf(label, callback, select=select)

    def add_sfinder(self, label, callback, select=False):
        self.display.add_sf(label, callback, select=select)

    def _opensgf(self):
        self._pause(True)
        super(ControllerV, self)._opensgf()
        self._pause(False)

    def _openvideo(self):
        self._pause(True)
        vidfile = self.display.promptopen()
        if len(vidfile):
            self.video = vidfile
        self._pause(False)

    def _openlive(self):
        self.video = 0

    def _save(self):
        self._pause(True)
        super(ControllerV, self)._save()
        self._pause(False)

    def __setattr__(self, name, value):

        # watch "current move number" field, and stop vision when user is browsing previous moves.
        if name == "current_mn" and value is not None:
            try:
                if value < self.kifu.last_move().number:
                    self._pause(True)  # don't update Pause object here
                else:
                    self._pause(self.paused.__nonzero__())
            except AttributeError:
                pass  # most likely last_move() has returned null

        return super(ControllerV, self).__setattr__(name, value)


class ControllerVSeq(ControllerBase):
    """
    Controller with no GUI that is supposed to be run in a single-threaded environment.

    """

    def __init__(self, kifufile=None, video=0, bounds=(0, 1)):
        super(ControllerVSeq, self).__init__(kifufile=kifufile)
        self.video = video
        self.bounds = bounds

    def pipe(self, instruction, args):
        """
        Execute command straight away (assumption of single-threaded environment).

        """
        self.api[instruction](*args)


class Pause(object):

    def __init__(self, paused=False):
        self.paused = paused

    def true(self):
        self.paused = True
        return self.paused

    def false(self):
        self.paused = False
        return self.paused

    def __nonzero__(self):
        return self.paused
