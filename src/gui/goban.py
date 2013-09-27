from go.kifu import Kifu
from config import *
from Tkinter import Tk, Canvas


__author__ = 'Kohistan'


class Goban():

    """
    Class modeling the GUI, that is to say a goban.
    """

    def __init__(self, kifu):
        self.kifu = kifu
        self._structure()
        self._draw()
        self._bind()

    def coord(self, move):

        """
        Returns a tuple of integer describing the row and column of the move.
        """
        row = move[2]
        col = move[3]
        a = ord(row) - 97
        b = ord(col) - 97
        return a, b

    def _structure(self):
        # the occupied positions storage
        self.grid = []
        for i in range(19):
            row = []
            for j in range(19):
                # empty positions matrix representing the goban grid
                row.append(["E", -42])
            self.grid.append(row)
        # mark the liberties that have been counted
        self.libs = []
        for i in range(19):
            row = []
            for j in range(19):
                # the first int is the chain id of the stone,
                # the second int is the Tk id of the stone,
                row.append(-1)
            self.libs.append(row)
        self.markid = 0
        self.deleted = []
        self.highlight_id = -42

    def _draw(self):

        """
        Draw an empty goban.
        """
        master = Tk()
        self.canvas = Canvas(master, width=gsize * rwidth, height=gsize * rwidth)
        self.canvas.pack()
        self.border = 3
        self.canvas.configure(background="#F0CAA7")
        # vertical lines
        offset = rwidth / 2
        for i in range(gsize):
            x = i * rwidth + offset
            self.canvas.create_line(x, offset, x, gsize*rwidth-offset)
            # horizontal lines
        for i in range(gsize):
            y = i * rwidth + offset
            self.canvas.create_line(offset, y, gsize*rwidth-offset, y)
            # hoshis
        for a in [3, 9, 15]:
            wid = 3
            for b in [3, 9, 15]:
                xcenter = a * rwidth + rwidth / 2
                ycenter = b * rwidth + rwidth / 2
                oval = self.canvas.create_oval(xcenter-wid, ycenter-wid, xcenter+wid, ycenter+wid)
                self.canvas.itemconfigure(oval, fill="black")

    def _bind(self):

        """
        Bind the action listeners.
        """
        self.canvas.bind("<Button-1>", self._play)
        self.canvas.bind("<Button-2>", self._backward)
        # the canvas needs the focus to listen the keyboard
        self.canvas.focus_set()
        self.canvas.bind("<Right>", self._forward)
        self.canvas.bind("<Up>", self._forward)
        self.canvas.bind("<Left>", self._backward)
        self.canvas.bind("<Down>", self._backward)

    def _play(self, event):

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
            move = color + "[" + row + col + "]"
            self.kifu.move(move)
            self._stone(self.kifu.current.value)
            self._highlight(self.kifu.current.value)

    def _forward(self, event):
        """
        Internal function to display the next kifu stone on the goban.
        """
        if 0 < len(self.kifu.current.children):
            current = self.kifu.current
            self.kifu.current = current.children[len(current.children)-1]
            self._stone(self.kifu.current.value)
            self._highlight(self.kifu.current.value)

    def _backward(self, event):

        """
        Internal function to undo the last move made on the goban.
        """
        current = self.kifu.current
        if current.value != "root":
            (a, b) = self.coord(current.value)
            self.canvas.delete(self.grid[a][b][1])
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
        oval_id = self.canvas.create_oval(x0, y0, x1, y1)
        color = "black"
        if move[0] == 'W':
            color = "white"
        self.canvas.itemconfigure(oval_id, fill=color)
        self.grid[a][b][0] = color[0].upper()
        self.grid[a][b][1] = oval_id
        self._checklibs(move)

    def _highlight(self, move):
        self.canvas.delete(self.highlight_id)
        (a, b) = self.coord(move)
        x0 = a * rwidth + 5 * self.border
        y0 = b * rwidth + 5 * self.border
        x1 = (a + 1) * rwidth - 5 * self.border
        y1 = (b + 1) * rwidth - 5 * self.border
        if move[0] == 'B':
            colo = "white"
        else:
            colo = "black"
        self.highlight_id = self.canvas.create_oval(x0, y0, x1, y1)
        self.canvas.itemconfigure(self.highlight_id, fill=colo)

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
        self.canvas.delete(self.grid[a][b][1])
        color = self.grid[a][b][0]
        captured.append(color + "[" + chr(a+97) + chr(b+97) + "]")
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

    def display(self):

        """
        Call the main loop of Tk.
        """

        self.canvas.mainloop()


if __name__ == '__main__':
    kifu = Kifu.parse("/Users/Kohistan/Documents/Go/Legend Games/MilanMilan-Korondo.sgf")
    goban = Goban(kifu)
#    vision = Vision(goban)
#    vision.start()
    goban.display()
