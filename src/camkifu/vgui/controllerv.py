from queue import Full, Empty, Queue

from camkifu.core.exceptions import PipeWarning
from golib.config.golib_conf import B, W, E
from golib.gui.controller import Controller


__author__ = 'Arnaud Peloquin'

commands_size = 10


def promptdiscard(meth):
    """
    ControllerV Method decorator. Prompt the user with a "discard changes / cancel" dialog when doing an operation
    that may compromise unsaved changes in the current game record (eg. changing video input).

    """
    def wrapper(*args, **kwargs):
        self = args[0]
        self._pause(True)
        if self.kifu.modified and self.display.promptdiscard(title="Discard current changes ?"):
            # the "modified" flag may have to be checked by other methods in the call stack,
            # so remember that choice in order not to prompt the user multiple times.
            self.kifu.modified = False
        if not self.kifu.modified:
            meth(*args, **kwargs)
        self._pause(False)
    return wrapper


# noinspection PyMethodMayBeStatic
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
            self.input.commands["next"] = lambda: self.next()  # lambda needed to bind method externally at runtime
            self.input.commands["on"] = lambda: self._on()  # lambda needed (same reason)
            self.input.commands["off"] = lambda: self._off()  # lambda needed (same reason)
            self.input.commands["run"] = self._run
            self.input.commands["pause"] = lambda: self._pause(self.paused.true())
            self.input.commands["vidfile"] = self._openvideo
            self.input.commands["vidlive"] = self._openlive
        except AttributeError as ae:
            self.log("Some commands could not be bound to User Interface.")
            self.log(ae)

        self.api = {
            "append": self._cvappend,
            "delete": self._cvdelete,
            "bulk": self._cv_bulk,
            "register_bf": self._add_bfinder,
            "register_sf": self._add_sfinder,
            "select_bf": self._select_bfinder,
            "select_sf": self._select_sfinder,
            "video_changed": self._prompt_new_kifu
        }

        if sgffile is not None:
            self._goto(722)  # get kifu ready to ramble

        self.paused = Pause()

    def pipe(self, instruction, args=None):
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
                    if args is not None:
                        self.api[instruction](*args)
                    else:
                        self.api[instruction]()
                except KeyError:
                    pass  # instruction not implemented here
                except Exception as e:
                    print("Instruction [%s] with arguments [%s] did not complete normally" % (instruction, args))
                    raise e
        except Empty:
            pass

    def locate(self, x, y):
        """
        Look for a Move object having (x, y) location in the kifu.

        x, y -- interpreted in the opencv (=tk) coordinates frame.

        """
        node = self.kifu.locate(x, y)
        if node is not None:
            return node.getmove()

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

    def next(self):
        """
        To be set from outside (eg. by Vision Manager).
        The user has clicked "Next".

        """
        pass

    def _run(self):
        """
        Ask vision processes to run / resume.

        """
        if self.at_last_move():
            self._pause(self.paused.false())
        else:
            self.log("Processing can't create variations in game. Please navigate to the last move.")

    def _add_bfinder(self, bf_class, callback):
        self.display.add_bf(bf_class, callback)

    def _add_sfinder(self, label, callback):
        self.display.add_sf(label, callback)

    def _select_bfinder(self, label):
        self.display.select_bf(label)

    def _select_sfinder(self, label):
        self.display.select_sf(label)

    def _prompt_new_kifu(self):
        """
        To be called when the current Kifu must imperatively be replaced.
        For example, if the video input has changed, there isn't much sense in keeping the same SGF.

        """
        if not self._opensgf():
            self._newsgf()

    def _cvappend(self, move):
        """
        Append the provided move to the current game. Implementation dedicated to automated detection.

        """
        move.number = self.current_mn + 1
        self.rules.put(move)
        self._append(move)

    def _cvdelete(self, x, y):
        """
        Remove the move at the given position from the current game. Implementation dedicated to automated
        detection.

        Raises exception if there is no move present at that location.

        """
        with self.rlock:
            self.selected = x, y
            self._delete(learn=False)  # don't send info back to algorithm since it (supposedly) comes from it already

    def _cv_bulk(self, moves):
        """
        Bulk update of the goban with multiple moves, allowing for better performance (up to 2 updates only of GUI and
        structures for the whole bulk).

        moves -- the list of moves to apply. those may be of color 'E', then meaning the removal of the
        stone already present (the thread applying the update will most likely complain if no stone is present).

        """
        order = {E: 0, B: 1, W: 1}  # keep B/W order, but removals must be performed and confirmed before appending
        moves = sorted(moves, key=lambda m: order[m.color])
        # todo this is currently being run on main thread. see to use a worker thread instead ?
        with self.rlock:
            if self.at_last_move():
                self.rules.reset()
                i = 0
                mv = moves[i]
                while mv.color is E:
                    torem = self.kifu.locate(mv.x, mv.y).getmove()
                    self.rules.remove(torem, reset=False)
                    self.kifu.delete(torem)
                    self.current_mn -= 1
                    i += 1
                    if i < len(moves):
                        mv = moves[i]
                    else: break
                if i:
                    self.rules.confirm()  # save the changes and update GUI
                while i < len(moves):
                    assert mv.color in (B, W)
                    mv.number = self.current_mn + 1
                    self.rules.put(mv, reset=False)
                    self.kifu.append(mv)
                    self.current_mn += 1
                    i += 1
                    if i < len(moves):
                        mv = moves[i]
                    else: break
                self.rules.confirm()  # save the changes and update GUI
                self.log_mn()
            else:
                raise NotImplementedError("Variations not allowed yet. Please navigate to end of game.")

    def _mouse_release(self, event):
        """ Method override to add correction event. """
        move = super(ControllerV, self)._mouse_release(event)
        if move is not None:
            self.corrected(None, move)

    def _delete(self, event=None, learn=True):
        """
        Method override to introduce a correction event, passed down to stones finder.

        learn -- True to pass this correction down to the listeners, False to do it silently.

        """
        move = super(ControllerV, self)._delete(event)
        if learn and move is not None:
            self.corrected(move, None)

    def _drag(self, event):
        """ Method override to add correction event. """
        moves = super(ControllerV, self)._drag(event)
        if moves is not None:
            self.corrected(*moves)

    def _opensgf(self):
        """ Method override to pause vision threads during long GUI operations. """
        self._pause(True)
        opened = super(ControllerV, self)._opensgf()
        self._pause(False)
        return opened

    def _save(self):
        """ Method override to pause vision threads during long GUI operations. """
        self._pause(True)
        super(ControllerV, self)._save()
        self._pause(False)

    @promptdiscard
    def _openvideo(self):
        """
        Change the video source to a file selected by user, that should be processed by detection algorithms.
        This is likely to discard the previous video source being processed (up to the video manager to take that
        change into account).

        """
        vidfile = self.display.promptopen()
        if len(vidfile):
            self.video = vidfile
            self.bounds = (0, 1)  # reset to default bounds (read entire file)

    @promptdiscard
    def _openlive(self):
        """
        Open the live camera, that should be processed by detection algorithms.
        This is likely to discard the previous video source being processed.

        """
        self.video = 0

    def __setattr__(self, name, value):
        """ Method override to pause vision threads when browsing previous moves. """

        # watch "current move number" field, and stop vision when user is browsing previous moves.
        if name == "current_mn" and value is not None:
            try:
                if value < self.kifu.lastmove().number:
                    self._pause(True)  # don't update Pause object here
                else:
                    self._pause(self.paused.__bool__())
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