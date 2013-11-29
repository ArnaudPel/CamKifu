from Queue import Queue, Full, Empty
from Tkinter import Tk
from threading import RLock
from config.guiconf import rwidth
from go.kifu import Kifu

from go.rules import Rule
from go.sgf import Move
from gui.pipewarning import PipeWarning
from gui.ui import UI


__author__ = 'Kohistan'


class ControllerUnsafe(object):
    """
    Class arbitrating the interactions between user input, vision input, and display.

    """

    def __init__(self, kifu, user_input, display):
        self.kifu = kifu
        self.rules = Rule()
        self.current_mn = 0

        self.queue = Queue(10)
        self.api = {
            "append": lambda c, x, y: self._put(Move(c, x, y), method=self._append)
        }

        self.display = display
        self.input = user_input
        self.clickloc = None
        self._bind()

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

    def _bind(self):

        """
        Bind the action listeners.
        """
        self.input.mousein.bind("<Button-1>", self._click)
        self.input.mousein.bind("<B1-Motion>", self._drag)
        self.input.mousein.bind("<ButtonRelease-1>", self._mouse_release)
        self.input.mousein.bind("<Button-2>", self._backward)

        self.input.keyin.bind("<Right>", self._forward)
        self.input.keyin.bind("<Up>", self._forward)
        self.input.keyin.bind("<Left>", self._backward)
        self.input.keyin.bind("<Down>", self._backward)
        self.input.keyin.bind("<p>", self.printself)
        self.input.keyin.bind("<Escape>", lambda _: self.display.select(None))

        # commands from background that have to be executed on the GUI thread.
        self.input.bind("<<execute>>", self._execute)

        # dependency injection attempt
        try:
            self.input.commands["save"] = self.save
        except AttributeError:
            print "Some commands could not be bound to User Interface."

    def _click(self, event):

        """
        Internal function to add a move to the kifu and display it. The move
        is expressed via a mouse click.
        """
        x = event.x / rwidth
        y = event.y / rwidth
        self.clickloc = (x, y)
        self.display.select(Move("Dummy", x, y))

    def _mouse_release(self, event):
        x_ = event.x / rwidth
        y_ = event.y / rwidth
        if (x_, y_) == self.clickloc:
            move = Move(self.kifu.next_color(), x_, y_)
            self._put(move, method=self._append)

    def _forward(self, event):
        """
        Internal function to display the next kifu stone on the goban.
        """
        lastmove = self.kifu.game.lastmove()
        if lastmove and (self.current_mn < lastmove.number):
            move = self.kifu.game.getmove(self.current_mn + 1).getmove()
            self._put(move, method=self.incr_move_number)

    def _put(self, move, method=None, highlight=True):
        allowed, data = self.rules.put(move)
        if allowed:
            if method is not None:
                # executed method before display and confirm, not to display anything in case of exception
                method(move)
            self.rules.confirm()
            self.display.display(move)
            if highlight:
                self.display.highlight(move)
            self.display.erase(data)
        else:
            print data

    def _append(self, move):
        last_move = self.kifu.game.lastmove()
        if not last_move or (self.current_mn == last_move.number):
            self.kifu.append(move)
            self.current_mn += 1
        else:
            raise NotImplementedError("Cannot create variations for a game yet. Sorry.")

    def _backward(self, event):

        """
        Internal function to undo the last move made on the goban.
        """
        if 0 < self.current_mn:
            move = self.kifu.game.getmove(self.current_mn).getmove()
            self._remove(move, method=self._prev_highlight)

    def _remove(self, move, method=None):
        allowed, details = self.rules.remove(move)
        if allowed:
            self.rules.confirm()
            self.display.erase(move)
            self.display.display(details)  # put previously dead stones back
            if method is not None:
                method(move)
        else:
            print details

    def _prev_highlight(self, _):
        self.current_mn -= 1
        if 0 < self.current_mn:
            prev_move = self.kifu.game.getmove(self.current_mn).getmove()
            self.display.highlight(prev_move)
        else:
            self.display.highlight(None)

    def _drag(self, event):
        x_ = event.x / rwidth
        y_ = event.y / rwidth
        if self.clickloc != (x_, y_):
            color = self.rules.stones[self.clickloc[0]][self.clickloc[1]]
            if color in ('B', 'W'):
                origin = Move(color, *self.clickloc)
                dest = Move(color, x_, y_)
                rem_allowed, freed = self.rules.remove(origin)
                if rem_allowed:
                    put_allowed, captured = self.rules.put(dest, reset=False)
                    if put_allowed:
                        self.rules.confirm()
                        self.kifu.relocate(origin, dest)
                        self.display.relocate(origin, dest)
                        self.display.display(freed)
                        self.display.erase(captured)
                        self.clickloc = x_, y_

    def save(self):
        if self.kifu.sgffile is not None:
            self.kifu.save()
        else:
            sfile = self.display.promptsave()
            if sfile is not None:
                self.kifu.sgffile = sfile
                self.kifu.save()
            else:
                print "Saving cancelled."

    def incr_move_number(self, _):
        self.current_mn += 1

    def printself(self, event):
        print self.rules


class Controller(ControllerUnsafe):

    def _put(self, move, method=None, highlight=True):
        with RLock():
            super(Controller, self)._put(move, method, highlight)

    def _remove(self, move, method=None):
        with RLock():
            super(Controller, self)._remove(move, method)


if __name__ == '__main__':
    root = Tk()
    #kifu = Kifu.parse("/Users/Kohistan/Documents/go/Perso Games/MrYamamoto-Kohistan.sgf")
    kifu = Kifu.new()

    app = UI(root)
    control = ControllerUnsafe(kifu, app, app)
    root.mainloop()