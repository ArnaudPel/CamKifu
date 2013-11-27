from Queue import Queue, Full, Empty
from Tkinter import Tk
from go.kifu import Kifu

from go.rules import Rule
from go.sgf import Move
from gui.pipewarning import PipeWarning
from gui.ui import UI


__author__ = 'Kohistan'


class Controller():
    """
    Class arbitrating the interactions between user input, vision input, and display.

    """

    def __init__(self, kifu, bound, display):
        self.kifu = kifu
        self.rules = Rule()
        self.current_mn = 0

        self.queue = Queue(10)
        self.api = {"add": self._append}

        self.display = display
        self.bound = bound
        self._bind()

    def pipe(self, instruction, args):
        #if self.bound.closed:
        #    raise PipeWarning("Goban has been closed")
        if instruction == "event":
            self.bound.event_generate(args)
        else:
            try:
                self.queue.put_nowait((instruction, args))
            except Full:
                print "Goban instruction queue full, ignoring {0}".format(instruction)
            self.bound.event_generate("<<execute>>")

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
        self.bound.bind("<Button-1>", self._click)
        self.bound.bind("<Button-2>", self._backward)

        # todo remove if not needed
        # the canvas needs the focus to listen the keyboard
        self.bound.focus_set()

        self.bound.bind("<Right>", self._forward)
        self.bound.bind("<Up>", self._forward)
        self.bound.bind("<Left>", self._backward)
        self.bound.bind("<Down>", self._backward)
        self.bound.bind("<p>", self.printself)

        # virtual commands
        self.bound.bind("<<execute>>", self._execute)

    def _click(self, event):

        """
        Internal function to add a move to the kifu and display it. The move
         is expressed via a mouse click.
        """
        x_ = event.x / 40
        y_ = event.y / 40
        move = Move(self.kifu.next_color(), x_, y_)
        allowed, data = self.rules.next(move)
        if allowed:
            self.rules.confirm()
            self._append(x_, y_, move.color)
            self.display.erase(data)
        else:
            print data

    def _forward(self, event):
        """
        Internal function to display the next kifu stone on the goban.
        """
        if self.current_mn < self.kifu.game.lastmove():
            self.current_mn += 1
            move = self.kifu.game.getmove(self.current_mn).getmove()
            allowed, data = self.rules.next(move)
            if allowed:
                self.rules.confirm()
                self.display.display(move)
                self.display.highlight(move)
                self.display.erase(data)
            else:
                print data

    def _backward(self, event):

        """
        Internal function to undo the last move made on the goban.
        """
        if 0 < self.current_mn:
            move = self.kifu.game.getmove(self.current_mn).getmove()
            allowed, details = self.rules.previous(move)
            if allowed:
                self.rules.confirm()
                self.display.erase([move])
                self.current_mn -= 1
                if 0 < self.current_mn:
                    prev_move = self.kifu.game.getmove(self.current_mn).getmove()
                    self.display.highlight(prev_move)
                for move in details:
                    self.display.display(move)  # put previously dead stones back
            else:
                print details

    def _append(self, x, y, color):
        if self.current_mn == self.kifu.game.lastmove():
            move = Move(color, x, y)
            self.kifu.append(move)
            self.current_mn += 1
            self.display.display(move)
            self.display.highlight(move)
        else:
            raise NotImplementedError("Cannot create variations for a game yet. Sorry.")

    def printself(self, event):
        print self.rules

if __name__ == '__main__':
    root = Tk()
    #kifu = Kifu.parse("/Users/Kohistan/Documents/go/Perso Games/MrYamamoto-Kohistan.sgf")
    kifu = Kifu.new()

    app = UI(root)
    control = Controller(kifu, app.goban, app.goban)
    root.mainloop()