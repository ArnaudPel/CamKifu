import os
import queue
import random

import numpy as np

import camkifu.core
import golib.gui
import golib.model
from camkifu.config.cvconf import tmp_sgf
from golib.config.golib_conf import gsize, E, B, W

# size of the command queue
commands_size = 10


def promptdiscard(meth):
    """ ControllerV Method decorator. Prompt the user with a "discard changes / cancel" dialog.
    Use when doing an operation that may compromise unsaved changes in the game record (eg. changing video input).

    Arg:
        meth: callable
            The method to decorate.
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
class ControllerV(golib.gui.Controller):
    # noinspection PyUnresolvedReferences
    """ Extension of the default GUI controller adding the handling of Vision threads.

        Attributes:
            queue: Queue
                The commands that have been received and wait for execution.
            video: int / str
                The video input description as used by openCV: integer or file path
            bounds: (int, int)
                The video start and end position as used by openCV.
            input:
                The UI component collecting user input.
            api: dict
                The commands exposed by this Controller.
            paused: Paused
                A toggle
        """

    def __init__(self, user_input, display, sgffile=None, video=0, bounds=(0, 1)):
        sgf, autosaved = self._init_sgf(sgffile)
        super().__init__(user_input, display, sgffile=sgf)
        self.queue = queue.Queue(commands_size)
        self.video = video
        self.bounds = bounds

        # ensure the commands are executed by the proper GUI thread
        self.input.bind("<<execute>>", self._cmd)

        try:
            # setup the UI response to known commands
            self.input.commands["next"] = lambda: self.next()  # lambda needed to bind method externally at runtime
            self.input.commands["on"] = lambda: self._on()     # lambda needed (same reason)
            self.input.commands["off"] = lambda: self._off()   # lambda needed (same reason)
            self.input.commands["run"] = self._run
            self.input.commands["pause"] = lambda: self._pause(self.paused.true())
            self.input.commands["vidfile"] = self._openvideo
            self.input.commands["vidlive"] = self._openlive
            self.input.commands["vidpos"] = lambda new_pos: self.vidpos(new_pos)
            self.input.commands["snapshot"] = lambda save_goban: self.snapshot(save_goban)
            self.input.commands["random"] = lambda: self.random()
            self.input.commands["quit"] = lambda: self.quit()
        except AttributeError as ae:
            self.log("Some commands could not be bound to User Interface.")
            self.log(ae)

        self.api = {
            # expose more commands for video manager / vision threads
            "append": self._cvappend,
            "delete": lambda x, y: self._delete(y, x, learn=False),  # NB: (x,y) inversion ! Call with numpy coordinates
            "bulk": self._bulk_update,
            "register_bf": self._add_bfinder,
            "register_sf": self._add_sfinder,
            "select_bf": self._select_bfinder,
            "select_sf": self._select_sfinder,
            "video_changed": self._newsgf,
            "video_progress": lambda progress: self.display.video_progress(progress),
            "auto_save": lambda: self.auto_save(),
        }
        if autosaved:
            # abstraction leak ! :)
            self.kifu.sgffile = None
            self.kifu.modified = True  # recovered from a crash, treat as a new game
            self.display_title("Recovered from auto-saved game")
        if sgf is not None:
            self.goto(722)  # get kifu ready to ramble

        self.paused = Pause(False)

    def _init_sgf(self, sgffile):
        sgf = sgffile
        autosaved = False
        if sgffile is None and os.path.exists(tmp_sgf):
            sgf = tmp_sgf
            autosaved = True
        return sgf, autosaved

    def pipe(self, instruction, *args):
        """ Send an instruction to this controller, that will be treated asynchronously.

        Args:
            instruction: callable
            args: the arguments to pass to "instruction".
        """
        if self.input.closed:
            raise camkifu.core.ControllerWarning("Target User Interface has been closed.")
        if instruction == "event":
            # virtual event, comes from self.input itself, neither keyin nor mousein
            self.input.event_generate(*args)
        else:
            try:
                self.queue.put_nowait((instruction, args))
            except queue.Full:
                print("Controller instruction queue full, ignoring {0}".format(instruction))
            self.input.event_generate("<<execute>>")

    def _cmd(self, event):
        """ Try to empty the commands queue.

        This method will not execute more than a fixed number of commands, in order to prevent infinite looping.
        Such infinite looping could occur if this method fails to keep up with other threads piping commands.
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
        except queue.Empty:
            pass

    def get_stones(self) -> np.ndarray:
        """ Return a copy of the current goban state, in the numpy coordinates system.
        """
        with self.rlock:
            return np.array(self.rules.stones, dtype=object).T

    def _on(self):
        """ Turn on vision processing if it is not running. To be set from outside (eg. by Vision Manager).
        """
        pass

    def _off(self):
        """ Turn off vision processing if it is running. To be set from outside (eg. by Vision Manager).
        """
        pass

    def _pause(self, boolean):
        """ Pause if boolean is True, else resume. To be set from outside (eg. by Vision Manager).
        """
        pass

    def corrected(self, err_move, exp_move):
        """ The user has made a manual modification to the Goban. To be set from outside (eg. by Vision Manager).

        Args:
            err_move: Move
                The move that was deleted by the user. May be null if the correction is an add.
            exp_move: Move
                The move that was added by the user. May be null if the correction is a delete.
        """
        pass

    def next(self):
        """ The user has clicked "Next". To be set from outside (eg. by Vision Manager).
        """
        pass

    def vidpos(self, new_pos):
        """ The user has set a new video position. To be set from outside (eg. by Vision Manager).

        Arg:
            new_pos: float
                The video progress to set (in %).
        """
        pass

    def snapshot(self, save_goban):
        """ Take a screenshot and save it to filesystem. To be set from outside (eg. by Vision Manager).
        """
        pass

    def random(self):
        """ Randomly fill the goban with stones (for neural network training).
            Respect existing stones.
        """
        moves = []
        rand = random.Random()
        stones = self.get_stones()
        unique, counts = np.unique(stones.flatten(), return_counts=True)
        density = (gsize ** 2 - counts[np.where(unique == E)[0][0]]) / (gsize ** 2)
        if 0.8 < density:
            print("Goban already too dense, will not randomly fill it more.")
            return
        upbound = 2 + int(10 * density)
        for r in range(gsize):
            for c in range(gsize):
                if stones[r, c] == E:
                    p = rand.randint(0, upbound)
                    if upbound - 2 < p:
                        color = B if p == upbound - 1 else W
                        moves.append(golib.model.Move('np', (color, r, c)))
        self._bulk_update(moves)

    def _run(self):
        """ Ask vision processes to run / resume.
        """
        if self.at_last_move():
            self._pause(self.paused.false())
        else:
            self.log("Processing can't create variations in game. Please navigate to the last move.")

    def _add_bfinder(self, bf_class, callback):
        self.display.add_bf(bf_class, callback)

    def _add_sfinder(self, sf_class, callback):
        self.display.add_sf(sf_class, callback)

    def _select_bfinder(self, label):
        self.display.select_bf(label)

    def _select_sfinder(self, label):
        self.display.select_sf(label)

    def _cvappend(self, move):
        """ Append the provided move to the current game. Implementation dedicated to automated detection.
        """
        move.number = self.head + 1
        self.rules.put(move)
        self._append(move)

    def _mouse_release(self, event):
        """ Method override to introduce a correction event.
        """
        move = super()._mouse_release(event)
        if move is not None:
            self.corrected(None, move)

    def _delete(self, x: int, y: int, learn: bool=True):
        """ Method override to introduce a correction event.

        Arg:
            learn: bool
                True to pass this correction down to the listeners, False to do it silently.
        """
        move = super()._delete(x, y)
        if learn and move is not None:
            self.corrected(move, None)
        return move

    def _drag(self, event):
        """ Method override to introduce a correction event.
        """
        moves = super()._drag(event)
        if moves is not None:
            self.corrected(*moves)

    def _opensgf(self):
        """ Method override to pause vision threads during long GUI operations.
        """
        self._pause(True)
        opened = super()._opensgf()
        self._pause(False)
        return opened

    def _save(self):
        """ Method override to pause vision threads during long GUI operations.
        """
        self._pause(True)
        super()._save()
        self.del_autosave()
        self._pause(False)

    @promptdiscard
    def _openvideo(self):
        """ Change the video source to a file selected by user.
        """
        vidfile = self.display.promptopen(title="Open video or image")
        if len(vidfile):
            self.video = vidfile
            self.bounds = (0, 1)  # reset to default bounds (read entire file)

    @promptdiscard
    def _openlive(self):
        """ Change the video source to the live camera.
        """
        self.video = 0

    def __setattr__(self, name, value):
        """ Method override to pause vision threads when browsing previous moves.
        """

        # watch "current move number" field, and stop vision when user is browsing previous moves.
        if name == "head" and value is not None:
            try:
                if value < self.kifu.lastmove().number:
                    self._pause(True)  # don't update Pause object here
                else:
                    self._pause(self.paused)
            except AttributeError:
                pass  # most likely last_move() has returned null
        return super().__setattr__(name, value)

    def auto_save(self):
        self.kifu.snapshot(tmp_sgf)

    def del_autosave(self):
        if os.path.exists(tmp_sgf):
            os.remove(tmp_sgf)

    def _onclose(self):
        try:
            super()._onclose()
        except SystemExit as goodbye:
            self.del_autosave()
            raise goodbye

    def quit(self):
        self.del_autosave()


class Pause:
    """ A toggle that can be used in lambda functions.
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
