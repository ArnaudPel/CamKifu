from queue import Full, Empty, Queue

from camkifu.core.exceptions import PipeWarning
from golib.gui.controller import Controller


__author__ = 'Arnaud Peloquin'

commands_size = 10


class ControllerV(Controller):
    """
    Extension of the default GUI controller, adding the handling of Vision threads.

    """

    def __init__(self, user_input, display, sgffile=None, video=0, bounds=(0, 1)):
        super(ControllerV, self).__init__(user_input, display, sgffile=sgffile)
        self.queue = Queue(commands_size)
        self.video = video
        # todo provide a way to set these bounds from the GUI ?
        self.bounds = bounds

        # commands from background that have to be executed on the GUI thread.
        self.input.bind("<<execute>>", self._cmd)

        try:
            self.input.commands["on"] = lambda: self._on()  # lambda needed to enable external method biding at runtime
            self.input.commands["off"] = lambda: self._off()  # lambda needed (same reason)
            self.input.commands["run"] = self._run
            self.input.commands["pause"] = lambda: self._pause(self.paused.true())
            self.input.commands["vidfile"] = self._openvideo
            self.input.commands["vidlive"] = self._openlive
        except AttributeError as ae:
            self.log("Some commands could not be bound to User Interface.")
            self.log(ae)

        self.api = {
            "append": self.cvappend,
            "register_bf": self.add_bfinder,
            "register_sf": self.add_sfinder,
            "select_bf": self.select_bfinder,
            "select_sf": self.select_sfinder
        }

        if sgffile is not None:
            self._goto(722)  # get kifu ready to ramble

        self.paused = Pause()

    def pipe(self, instruction, args):
        """
        Send an instruction to this controller, that will be treated asynchronously.
        instruction -- a callable.
        args -- the arguments to to pass on "instruction" call.

        """
        if self.input.closed:
            raise PipeWarning("Target User Interface has been closed.")
        if instruction == "event":
            # virtual event, comes from self.input itself, neither keyin nor mousein
            self.input.event_generate(args)
        else:
            try:
                self.queue.put_nowait((instruction, args))
            except Full:
                print("Controller instruction queue full, ignoring {0}".format(instruction))
            self.input.event_generate("<<execute>>")

    def _cmd(self, event):
        """
        Try to empty the commands queue (filled by pipe()).
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
                except Exception as e:
                    print("Instruction [%s] with arguments [%s] did not complete normally" % (instruction, args))
                    raise e
        except Empty:
            pass

    def _on(self):
        """
        To be set from outside (eg. by Vision Manager).
        Turn on vision machinery if it is not running.

        """
        pass

    def _off(self):
        """
        To be set from outside (eg. by Vision Manager).
        Turn off vision machinery if it is running.

        """
        pass

    def _pause(self, boolean):
        """
        To be set from outside (eg. by Vision Manager).
        Pause if boolean is True, else resume.

        """
        pass

    def corrected(self, err_move, exp_move):
        """
        To be set from outside (eg. by Vision Manager).
        The user has made a manual modification to the Goban.

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

    def add_bfinder(self, bf_class, callback):
        self.display.add_bf(bf_class, callback)

    def add_sfinder(self, label, callback):
        self.display.add_sf(label, callback)

    def select_bfinder(self, label):
        self.display.select_bf(label)

    def select_sfinder(self, label):
        self.display.select_sf(label)

    def cvappend(self, move):
        """
        Append the provided move to the current game. Implementation dedicated to automated detection.

        """
        move.number = self.current_mn + 1
        self.rules.put(move)
        self._append(move)

    def _mouse_release(self, event):
        """ Method override to add correction event. """
        move = super(ControllerV, self)._mouse_release(event)
        if move is not None:
            self.corrected(None, move)

    def _delete(self, _):
        """ Method override to add correction event. """
        move = super(ControllerV, self)._delete(_)
        if move is not None:
            self.corrected(move, None)

    def _drag(self, event):
        """ Method override to add correction event. """
        moves = super(ControllerV, self)._drag(event)
        if moves is not None:
            self.corrected(*moves)

    def _opensgf(self):
        """ Method override to pause vision threads during long GUI operations. """
        self._pause(True)
        super(ControllerV, self)._opensgf()
        self._pause(False)

    def _openvideo(self):
        """
        Open a video file, that should be processed by detection algorithms.
        This is likely to discard the previous video source being processed.

        """
        self._pause(True)
        vidfile = self.display.promptopen()
        if len(vidfile):
            self.video = vidfile
            self.bounds = (0, 1)  # reset to default bounds (read entire file)
        self._pause(False)

    def _openlive(self):
        """
        Open the live camera, that should be processed by detection algorithms.
        This is likely to discard the previous video source being processed.

        """
        self.video = 0

    def _save(self):
        """ Method override to pause vision threads during long GUI operations. """
        self._pause(True)
        super(ControllerV, self)._save()
        self._pause(False)

    def __setattr__(self, name, value):
        """ Method override to pause vision threads when browsing previous moves. """

        # watch "current move number" field, and stop vision when user is browsing previous moves.
        if name == "current_mn" and value is not None:
            try:
                if value < self.kifu.lastmove().number:
                    self._pause(True)  # don't update Pause object here
                else:
                    self._pause(self.paused.__nonzero__())
            except AttributeError:
                pass  # most likely last_move() has returned null

        return super(ControllerV, self).__setattr__(name, value)


class Pause(object):
    """
    A toggle that can be used in lambda functions.

    """
    def __init__(self, paused=False):
        self.paused = paused

    def true(self):
        self.paused = True
        return self.paused

    def false(self):
        self.paused = False
        return self.paused

    def __bool__(self):
        return self.paused