from Tkinter import Canvas
import numpy as np
from config.guiconf import gsize, rwidth

__author__ = 'Kohistan'


class Goban(Canvas):
    def __init__(self, master):
        Canvas.__init__(self, master, width=gsize * rwidth, height=gsize * rwidth)
        self.stones = np.empty((gsize, gsize), dtype=np.object)
        self.closed = False
        self._draw_board()

    def _draw_board(self):
        """
        Draw an empty goban.

        """
        self.pack()
        self.configure(background="#F0CAA7")
        # vertical lines
        offset = rwidth / 2
        for i in range(gsize):
            x = i * rwidth + offset
            self.create_line(x, offset, x, gsize * rwidth - offset)
            # horizontal lines
        for i in range(gsize):
            y = i * rwidth + offset
            self.create_line(offset, y, gsize * rwidth - offset, y)
            # hoshis
        for a in [3, 9, 15]:
            wid = 3
            for b in [3, 9, 15]:
                xcenter = a * rwidth + rwidth / 2
                ycenter = b * rwidth + rwidth / 2
                oval = self.create_oval(xcenter - wid, ycenter - wid, xcenter + wid, ycenter + wid)
                self.itemconfigure(oval, fill="black")

    def display(self, moves, highlight=False):
        """
        Display a stone on the goban.

        """
        mves = moves if type(moves) in (list, set) else [moves]
        for move in mves:
            if self.stones[move.x][move.y] is not None:
                print "Warning: displaying a stone on top of another. Erasing previous stone."
                self.stones[move.x][move.y].erase()
            stone = Stone(self, move, highlight)
            stone.paint()
            self.stones[move.x][move.y] = stone

    def erase(self, moves):
        """
        coords -- the stones to erase from display.

        """
        mves = moves if type(moves) in (list, set) else [moves]
        for move in mves:
            self.stones[move.x][move.y].erase()  # clean canvas
            self.stones[move.x][move.y] = None

    def highlight(self, move, keep=False):
        if not keep:
            # loop is ugly, but no additional structure needed
            for stone in self:
                stone.highlight(False)
        if move:
            self.stones[move.x][move.y].highlight(True)

    def select(self, move):
        for stone in self:
            stone.select(False)
        try:
            self.stones[move.x][move.y].select(True)
        except AttributeError:
            pass  # selection cleared

    def __iter__(self):
        for x in range(gsize):
            for y in range(gsize):
                stone = self.stones[x][y]
                if stone is not None:
                    yield stone

    def relocate(self, origin, destination):
        stone = self.stones[origin.x][origin.y]
        if stone:
            stone.erase()
            self.stones[origin.x][origin.y] = None

            stone.setpos(destination.x, destination.y)
            stone.paint()
            self.stones[destination.x][destination.y] = stone
        else:
            print "Nothing to relocate."


tkcolors = {'B': "black", 'W': "white"}
tk_inv_colors = {'W': "black", 'B': "white"}


class Stone(object):
    def __init__(self, canvas, move, highlight=False, selected=False):
        self.canvas = canvas
        self._move = move.copy()  # self.move location may be changed by Stone
        self._hl = highlight
        self.selected = selected
        self.tkindexes = []
        self.border = 3

    def setpos(self, x, y):
        self._move.x = x
        self._move.y = y

    def paint(self):
        self.erase()  # clear any previous item from self
        self._paint_stone()
        self._paint_highlight()
        self._paint_selected()

    def erase(self):
        while len(self.tkindexes):
            idx = self.tkindexes.pop()
            self.canvas.delete(idx)

    def highlight(self, hl):
        if hl != self._hl:
            self._hl = hl
            self.erase()
            self.paint()

    def select(self, sel):
        if sel != self.selected:
            self.selected = sel
            self.erase()
            self.paint()

    def _paint_stone(self):
        x_ = self._move.x
        y_ = self._move.y

        x0 = x_ * rwidth + self.border
        y0 = y_ * rwidth + self.border
        x1 = (x_ + 1) * rwidth - self.border
        y1 = (y_ + 1) * rwidth - self.border

        oval_id = self.canvas.create_oval(x0, y0, x1, y1)
        self.canvas.itemconfigure(oval_id, fill=tkcolors[self._move.color])
        self.tkindexes.append(oval_id)

    def _paint_highlight(self):
        if self._hl:
            x_ = self._move.x
            y_ = self._move.y
            x0 = x_ * rwidth + 5 * self.border
            y0 = y_ * rwidth + 5 * self.border
            x1 = (x_ + 1) * rwidth - 5 * self.border
            y1 = (y_ + 1) * rwidth - 5 * self.border

            hl_id = self.canvas.create_oval(x0, y0, x1, y1)
            self.canvas.itemconfigure(hl_id, fill=tk_inv_colors[self._move.color])
            self.tkindexes.append(hl_id)

    def _paint_selected(self):
        if self.selected:
            x_ = self._move.x
            y_ = self._move.y

            x0 = x_ * rwidth
            y0 = y_ * rwidth
            x1 = (x_ + 1) * rwidth
            y1 = (y_ + 1) * rwidth

            oval_id = self.canvas.create_oval(x0, y0, x1, y1)
            self.canvas.itemconfigure(oval_id, outline="red", width=self.border)
            self.tkindexes.append(oval_id)

    def copy(self):
        return Stone(self.canvas, self._move, highlight=self._hl, selected=self.selected)





















