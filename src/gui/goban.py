from Queue import Queue, Full, Empty
from Tkinter import Tk, Canvas

from go.kifu import Kifu
from config.guiconf import *
from gui.cthread import AutoClick


__author__ = 'Kohistan'


class Goban():
    """
    Class modeling the GUI, that is to say a goban.
    """

    def __init__(self, root, kifu):
        self._root = root
        self._canvas = Canvas(root, width=gsize * rwidth, height=gsize * rwidth)
        self.border = 3
        self.closed = False

        self.kifu = kifu
        self.grid = []
        self.libs = []
        self.markid = 0
        self.deleted = []
        self.highlight_id = -42

        self._structure()
        self._draw()
        self._bind()

        self.queue = Queue(10)

    @staticmethod
    def coord(move):

        """
        Returns a tuple of integer describing the row and column of the move.
        move -- a string formatted on 5 characters as follow: "W[ab]"
                W is the color of the player
                a is the row
                b is the column
        """
        row = move[2]
        col = move[3]
        a = ord(row) - 97
        b = ord(col) - 97
        return a, b

    def pipe(self, instruction):
        try:
            self.queue.put_nowait(instruction)
        except Full:
            print "Goban instruction queue full, ignoring {0}".format(instruction)
        self._canvas.event_generate("<<execute>>")

    def _execute(self, event):
        try:
            while True:
                method, args = self.queue.get_nowait()
                method(*args)
        except Empty:
            pass

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

        # virtual commands
        self._canvas.bind("<<execute>>", self._execute)

    def _move(self, row, col, color):
        move = color + "[" + row + col + "]"
        self.kifu.move(move)
        self._stone(self.kifu.current.value)
        self._highlight(self.kifu.current.value)

    def _structure(self):
        # the occupied positions storage
        for i in range(19):
            row = []
            for j in range(19):
                # empty positions matrix representing the goban grid
                row.append(["E", -42])
            self.grid.append(row)
            # mark the liberties that have been counted
        for i in range(19):
            row = []
            for j in range(19):
                # the first int is the chain id of the stone,
                # the second int is the Tk id of the stone,
                row.append(-1)
            self.libs.append(row)

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

    def quit(self, event):
        self._root.quit()
        self.closed = True

    def _click(self, event):

        """
        Internal function to add a move to the kifu and display it. The move
         is expressed via a mouse click.
        """
        x_ = event.x / 40
        y_ = event.y / 40
        row = chr(97 + x_)
        col = chr(97 + y_)
        if self.grid[x_][y_][0] == "E":
            if self.kifu.current.value[0] == 'B':
                color = 'W'
            else:
                color = 'B'
            self._move(row, col, color)

    def _forward(self, event):
        """
        Internal function to display the next kifu stone on the goban.
        """
        if 0 < len(self.kifu.current.children):
            current = self.kifu.current
            self.kifu.current = current.children[len(current.children) - 1]
            self._stone(self.kifu.current.value)
            self._highlight(self.kifu.current.value)

    def _backward(self, event):

        """
        Internal function to undo the last move made on the goban.
        """
        current = self.kifu.current
        if current.value != "root":
            (a, b) = self.coord(current.value)
            self._canvas.delete(self.grid[a][b][1])
            self.grid[a][b] = ["E", -42]
            self.kifu.current = self.kifu.current.parent
            for move in self.deleted.pop():
                self._stone(move)
                # dirty fix because _stone() added a deleted element
                self.deleted.pop()
            self._highlight(self.kifu.current.value)

    def _stone(self, move):

        """
        Display a stone on the goban
        """
        (a, b) = self.coord(move)
        x0 = a * rwidth + self.border
        y0 = b * rwidth + self.border
        x1 = (a + 1) * rwidth - self.border
        y1 = (b + 1) * rwidth - self.border
        oval_id = self._canvas.create_oval(x0, y0, x1, y1)
        color = "black"
        if move[0] == 'W':
            color = "white"
        self._canvas.itemconfigure(oval_id, fill=color)
        self.grid[a][b][0] = color[0].upper()
        self.grid[a][b][1] = oval_id
        self._checklibs(move)

    def _highlight(self, move):
        self._canvas.delete(self.highlight_id)
        (a, b) = self.coord(move)
        x0 = a * rwidth + 5 * self.border
        y0 = b * rwidth + 5 * self.border
        x1 = (a + 1) * rwidth - 5 * self.border
        y1 = (b + 1) * rwidth - 5 * self.border
        if move[0] == 'B':
            colo = "white"
        else:
            colo = "black"
        self.highlight_id = self._canvas.create_oval(x0, y0, x1, y1)
        self._canvas.itemconfigure(self.highlight_id, fill=colo)

    def _checklibs(self, move):
        (a, b) = self.coord(move)
        color = move[0]
        # enemies libs to check first (attack advantage)
        enemies = []
        selfcheck = True
        for (i, j) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            row = a + i
            if row < 0 or 19 <= row: continue
            col = b + j
            if col < 0 or 19 <= col: continue
            othercolor = self.grid[row][col][0]
            if othercolor == "E":
                # no need to check own group libs if at least one liberty left
                selfcheck = False
                continue
            if othercolor != color:
                enemies.append((row, col))
        captured = []
        for coord in enemies:
            if not self._count(coord):
                self._clean(coord, captured)
                selfcheck = False
                # check for suicide play, undo if so
        if selfcheck:
            if not self._count((a, b)):
                print "suicide play"
                self._backward("Suicide Play")
                # store removed stones
        self.deleted.append(captured)

    def _count(self, (a, b)):

        """
        Recursively counts the group liberties, starting at the given position.
        """
        count = self._rcount((a, b), 0, self.markid)
        self.markid += 1
        return count

    def _rcount(self, (a, b), count, mid):
        color = self.grid[a][b][0]
        for (i, j) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            row = a + i
            if row < 0 or 19 <= row: continue
            col = b + j
            if col < 0 or 19 <= col: continue
            if self.grid[row][col][0] == "E" and self.libs[row][col] != mid:
                count += 1
                self.libs[row][col] = mid
            if self.grid[row][col][0] == color and self.libs[row][col] != mid:
                self.libs[row][col] = mid
                count += self._rcount((row, col), count, mid)
        return count

    def _clean(self, (a, b), captured):

        """
        Recursively removes the stone at the given position as well as all other
        connected stones
        """
        self._canvas.delete(self.grid[a][b][1])
        color = self.grid[a][b][0]
        captured.append(color + "[" + chr(a + 97) + chr(b + 97) + "]")
        self.grid[a][b] = ["E", -42]
        for (i, j) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            row = a + i
            if row < 0 or 19 <= row: continue
            col = b + j
            if col < 0 or 19 <= col: continue
            if self.grid[row][col][0] == color:
                self._clean((row, col), captured)

    def _printgrid(self):
        for i in range(19):
            line = ""
            for j in range(19):
                char = self.grid[j][i][0]
                if char == "E":
                    char = '-'
                line += char + "\t"
            print line
        print "--------------"


if __name__ == '__main__':
    kifu = Kifu.parse("/Users/Kohistan/Documents/Go/Legend Games/MilanMilan-Korondo.sgf")
    root = Tk()
    goban = Goban(root, kifu)
    autoplay = AutoClick(goban)
    autoplay.start()
    root.mainloop()
