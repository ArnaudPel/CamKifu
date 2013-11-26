import numpy as np

from Queue import Queue, Full, Empty
from Tkinter import Tk, Canvas

from go.kifu import Kifu
from config.guiconf import *
from go.rules import Rule, coord
from gui.pipewarning import PipeWarning


__author__ = 'Kohistan'


class GobanTk():
    """
    Class modeling the GUI, that is to say a goban.
    """

    def __init__(self, master, kifu):
        self.kifu = kifu
        self.rules = Rule()

        self._master = master
        self._canvas = Canvas(master, width=gsize * rwidth, height=gsize * rwidth)
        self.border = 3
        self.tkindexes = np.zeros((gsize, gsize), dtype=np.uint16)
        self.highlight_id = -1

        self.queue = Queue(10)
        self.api = {"add": self._move}
        self.closed = False

        self._draw()
        self._bind()

    def pipe(self, instruction, args):
        if self.closed:
            raise PipeWarning("Goban has been closed")
        if instruction == "event":
            self._canvas.event_generate(args)
        else:
            try:
                self.queue.put_nowait((instruction, args))
            except Full:
                print "Goban instruction queue full, ignoring {0}".format(instruction)
            self._canvas.event_generate("<<execute>>")

    def _execute(self, event):
        try:
            while True:
                instruction, args = self.queue.get_nowait()
                try:
                    self.api[instruction](*args)
                except KeyError:
                    pass  # instruction not implemented here
        except Empty:
            pass

    def quit(self, event):
        self._master.quit()
        self.closed = True

    def printself(self, event):
        print self.rules

    def _bind(self):

        """
        Bind the action listeners.
        """
        self._canvas.bind("<Button-1>", self._click)
        self._canvas.bind("<Button-2>", self._backward)
        # the canvas needs the focus to listen the keyboard
        self._canvas.focus_set()
        self._canvas.bind("<Right>", self._forward)
        self._canvas.bind("<Up>", self._forward)
        self._canvas.bind("<Left>", self._backward)
        self._canvas.bind("<Down>", self._backward)
        self._canvas.bind("<q>", self.quit)
        self._canvas.bind("<p>", self.printself)

        # virtual commands
        self._canvas.bind("<<execute>>", self._execute)

    def _draw(self):

        """
        Draw an empty goban.
        """
        self._canvas.pack()
        self._canvas.configure(background="#F0CAA7")
        # vertical lines
        offset = rwidth / 2
        for i in range(gsize):
            x = i * rwidth + offset
            self._canvas.create_line(x, offset, x, gsize * rwidth - offset)
            # horizontal lines
        for i in range(gsize):
            y = i * rwidth + offset
            self._canvas.create_line(offset, y, gsize * rwidth - offset, y)
            # hoshis
        for a in [3, 9, 15]:
            wid = 3
            for b in [3, 9, 15]:
                xcenter = a * rwidth + rwidth / 2
                ycenter = b * rwidth + rwidth / 2
                oval = self._canvas.create_oval(xcenter - wid, ycenter - wid, xcenter + wid, ycenter + wid)
                self._canvas.itemconfigure(oval, fill="black")

    def _click(self, event):

        """
        Internal function to add a move to the kifu and display it. The move
         is expressed via a mouse click.
        """
        x_ = event.x / 40
        y_ = event.y / 40
        row = chr(97 + x_)
        col = chr(97 + y_)
        color = 'W' if self.kifu.current.value[0] == 'B' else 'B'
        move = "{0}[{1}{2}]".format(color, row, col)
        allowed, data = self.rules.next(move)
        if allowed:
            self.rules.confirm()
            self._move(move)
            self._erase(data)
        else:
            print data

    def _forward(self, event):
        """
        Internal function to display the next kifu stone on the goban.
        """
        if 0 < len(self.kifu.current.children):
            move = self.kifu.current.children[-1].value
            allowed, data = self.rules.next(move)
            if allowed:
                self.rules.confirm()
                self._move(move)
                self._erase(data)
            else:
                print data

    def _move(self, move):
        self.kifu.move(move)
        self._display(self.kifu.current.value)
        self._highlight(self.kifu.current.value)

    def _backward(self, event):

        """
        Internal function to undo the last move made on the goban.
        """
        current = self.kifu.current
        if current.value != "root":
            allowed, details = self.rules.previous(current.value)
            if allowed:
                self.rules.confirm()
                (a, b) = coord(current.value)
                self._canvas.delete(self.tkindexes[a][b])
                self.kifu.current = self.kifu.current.parent
                self._highlight(self.kifu.current.value)
                for move in details:
                    self._display(move)  # put previously dead stones back
            else:
                print details

    def _display(self, move):

        """
        Display a stone on the goban
        """
        (a, b) = coord(move)
        color = "black" if move[0] == 'B' else "white"

        x0 = a * rwidth + self.border
        y0 = b * rwidth + self.border
        x1 = (a + 1) * rwidth - self.border
        y1 = (b + 1) * rwidth - self.border
        oval_id = self._canvas.create_oval(x0, y0, x1, y1)

        self._canvas.itemconfigure(oval_id, fill=color)
        self.tkindexes[a][b] = oval_id

    def _erase(self, coords):
        """
        coords -- a list of integer coordinates, the stones to erase from display

        """
        for move in coords:
            a, b = coord(move)
            self._canvas.delete(self.tkindexes[a][b])

    def _highlight(self, move):
        self._canvas.delete(self.highlight_id)
        (a, b) = coord(move)
        x0 = a * rwidth + 5 * self.border
        y0 = b * rwidth + 5 * self.border
        x1 = (a + 1) * rwidth - 5 * self.border
        y1 = (b + 1) * rwidth - 5 * self.border

        colo = "white" if move[0] == 'B' else "black"
        self.highlight_id = self._canvas.create_oval(x0, y0, x1, y1)
        self._canvas.itemconfigure(self.highlight_id, fill=colo)


if __name__ == '__main__':
    kifu = Kifu.parse("/Users/Kohistan/Documents/Go/Legend Games/MilanMilan-Korondo.sgf")
    root = Tk()
    goban = GobanTk(root, kifu)
    #autoplay = AutoClick(goban)
    #autoplay.start()
    root.mainloop()
