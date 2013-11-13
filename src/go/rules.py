from config.guiconf import gsize
from go.stateerror import StateError

__author__ = 'Kohistan'

"""
Hold the current state of a game, and ensure logical consistency of changes
made to that state.

"""

import numpy as np


class Rule(object):
    """
    This class is not thread safe.
    Its consistency is highly dependent on the good usage of self.confirm()

    """

    def __init__(self):
        self.stones = np.empty((gsize, gsize), dtype=np.str)
        self.stones.fill('E')
        self.last = None
        self.deleted = []
        self.lastdel = None  # a set of moves that have been deleted

    def confirm(self):
        """
        Persist the state of the last check, either next() or previous()

        """
        if self.last is not None:
            a, b = coord(self.last)
            color = self.last[0]
            self.stones[a][b] = color
            self.last = None
            if self.lastdel is not None:
                self.deleted.append(self.lastdel)
                for capt in self.lastdel:
                    x, y = coord(capt)
                    self.stones[x][y] = 'E'
                self.lastdel = None
            else:
                if len(self.deleted):
                    captured = self.deleted.pop()
                    if captured is not None:
                        for move in captured:
                            x, y = coord(move)
                            self.stones[x][y] = move[0]
        else:
            raise StateError("Confirming a forbidden state")

    def next(self, move):
        """
        Check if the move passed as argument can be performed.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for execution.

        """

        a, b = coord(move)
        color = move[0]
        enem_color = enemy(color)

        if self.stones[a][b] == 'E':
            self.stones[a][b] = color
            self.lastdel = set()

            # check if kill (attack advantage)
            enemies = []
            for row, col in connected(a, b):
                neighcolor = self.stones[row][col]
                if neighcolor == enem_color:
                    enemies.append((row, col))
            for x, y in enemies:
                group, nblibs = self._data(x, y)
                if nblibs == 0:
                    for k, l in group:
                        self.lastdel.add(getmove(enem_color, k, l))
                    self.last = move

                # check for suicide play if need be
            retval = True, self.lastdel
            if not self.last:
                _, nblibs = self._data(a, b)
                if not nblibs:
                    retval = False, "Suicide"
                    self.last = None
                else:
                    self.last = move

            # cancel
            self.stones[a][b] = 'E'
        else:
            retval = False, "Occupied"
            self.last = None

        return retval

    def previous(self, move):
        """
        Check if the move passed as argument can be undone.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for undo.

        """
        a, b = coord(move)
        allowed = self.stones[a][b] == move[0]
        if allowed:
            self.last = getmove('E', a, b)
            data = self.deleted[-1]
            self.lastdel = None
        else:
            self.last = None
            data = "Empty"
        return allowed, data

    def _data(self, a, b, _group=None, _libs=None):
        """
        Returns the list of stones and the number of liberties of the group at (a, b).

        a, b -- the coordinates of any stone of the group.
        _group, _libs -- internal variables used in the recursion, no need to set them from outside.

        """
        color = self.stones[a][b]
        if _group is None:
            assert color != 'E'
            _group = []
            _libs = []
        if (a, b) not in _group:
            _group.append((a, b))
            for x, y in connected(a, b):
                neighcolor = self.stones[x][y]
                if neighcolor == 'E':
                    if (x, y) not in _libs:
                        _libs.append((x, y))
                elif neighcolor == color:
                    self._data(x, y, _group, _libs)
        return _group, len(_libs)

    def __repr__(self):
        """
        For debugging purposes, can be modified at will.

        """
        string = ''
        for x in range(gsize):
            for y in range(gsize):
                string += self.stones[y][x]
                string += ' '
            string += "\n"
        return string


def connected(a, b):
    """
    Yields the (up to) 4 positions connected to (a, b).

    >>> [pos for pos in connected(0, 0)]
    [(1, 0), (0, 1)]
    >>> [pos for pos in connected(0, 5)]
    [(1, 5), (0, 6), (0, 4)]
    >>> [pos for pos in connected(5, 5)]
    [(4, 5), (6, 5), (5, 6), (5, 4)]
    >>> len([pos for pos in connected(gsize-1, gsize-1)])
    2
    >>> len([pos for pos in connected(gsize, gsize)])
    0

    """
    for (i, j) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        row = a + i
        if 0 <= row < gsize:
            col = b + j
            if 0 <= col < gsize:
                yield row, col


def coord(move):
    """
    Returns a tuple of integer describing the row and column of the move.
    move -- a string formatted on 5 characters as follow: "W[ab]"
            W is the color of the player
            a is the row
            b is the column

    >>> coord("W[aa]")
    (0, 0)
    >>> coord("W[cd]")
    (2, 3)
    >>> coord("B[ss]")
    (18, 18)
    """
    row = move[2]
    col = move[3]
    a = ord(row) - 97
    b = ord(col) - 97
    return a, b


def getmove(color, a, b):
    """
    >>> getmove('W', 0, 0)
    'W[aa]'
    >>> getmove('W', 2, 3)
    'W[cd]'
    >>> getmove('B', 18, 18)
    'B[ss]'
    """
    return color + "[" + chr(a + 97) + chr(b + 97) + "]"


def enemy(color):
    if color == 'B':
        return 'W'
    elif color == 'W':
        return 'B'
    else:
        raise ValueError("No enemy for '{0}'".format(color))











