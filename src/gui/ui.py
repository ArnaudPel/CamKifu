from Tkconstants import BOTH, LEFT, TOP
from tkFileDialog import asksaveasfilename
from Tkinter import Tk, Canvas, Misc
import traceback
from ttk import Frame, Button

import numpy as np

from config.guiconf import gsize, rwidth


__author__ = 'Kohistan'

"""
The main user interface.

"""


class UI(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.goban = Goban(self)
        self.init_ui()
        self.closed = False

        # user input part of the gui, delegated to goban ATM. may become lists later
        self.mousein = self.goban
        self.keyin = self.goban

        # these are expected to be set from outside, in an attempt to inject dependency via setter
        self.commands = {}

        # delegate some work to goban
        self.display = self.goban.display
        self.erase = self.goban.erase
        self.highlight = self.goban.highlight

    def init_ui(self):
        self.pack(fill=BOTH, expand=1)
        self.goban.pack(side=LEFT)

        b_save = Button(self, text="Save", command=self.save)
        b_pause = Button(self, text="Pause", command=self.pause)
        b_pause.pack(side=TOP, padx=5, pady=5)
        b_save.pack(side=TOP)

        self.goban.focus_set()
        self.goban.bind("<q>", self.close)  # dev utility mostly, will probably have to be removed

    def close(self, _):
        self.closed = True
        self.goban.closed = True
        Misc.quit(self)

    def save(self):
        self.execute("save")

    def pause(self):
        self.execute("pause")

    def execute(self, command):
        try:
            self.commands[command]()
        except KeyError:
            print "No \"{0}\" command set, ignoring.".format(command)
        except Exception:
            # keep going
            traceback.print_exc()
        self.goban.focus_set()

    # DISPLAY METHODS

    def promptsave(self):
        return asksaveasfilename(defaultextension=".sgf")


class Goban(Canvas):
    def __init__(self, master):
        Canvas.__init__(self, master, width=gsize * rwidth, height=gsize * rwidth)
        self.border = 3
        self.tkindexes = np.zeros((gsize, gsize), dtype=np.uint16)
        self.highlight_id = -1
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

    def display(self, move):
        """
        Display a stone on the goban.

        """
        x_ = move.x
        y_ = move.y
        color = "black" if move.color == 'B' else "white"

        x0 = x_ * rwidth + self.border
        y0 = y_ * rwidth + self.border
        x1 = (x_ + 1) * rwidth - self.border
        y1 = (y_ + 1) * rwidth - self.border
        oval_id = self.create_oval(x0, y0, x1, y1)

        self.itemconfigure(oval_id, fill=color)
        self.tkindexes[x_][y_] = oval_id

    def erase(self, moves):
        """
        coords -- the stones to erase from display.

        """
        for move in moves:
            self.delete(self.tkindexes[move.x][move.y])

    def highlight(self, move):
        """
        Delete the previous highlight circle, and draw a new highlight circle on the position.
        There's currently no check whether there's actually a stone at this position or not.

        """
        self.delete(self.highlight_id)
        if move is not None:
            x_ = move.x
            y_ = move.y
            x0 = x_ * rwidth + 5 * self.border
            y0 = y_ * rwidth + 5 * self.border
            x1 = (x_ + 1) * rwidth - 5 * self.border
            y1 = (y_ + 1) * rwidth - 5 * self.border

            colo = "white" if move.color == 'B' else "black"
            self.highlight_id = self.create_oval(x0, y0, x1, y1)
            self.itemconfigure(self.highlight_id, fill=colo)


