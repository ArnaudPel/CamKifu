from config.guiconf import gsize
from go.sgf import Move
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
            self.stones[self.last.x][self.last.y] = self.last.color
            self.last = None
            if self.lastdel is not None:
                self.deleted.append(self.lastdel)
                for capt in self.lastdel:
                    self.stones[capt.x][capt.y] = 'E'
                self.lastdel = None
            else:
                if len(self.deleted):
                    captured = self.deleted.pop()
                    if captured is not None:
                        for move in captured:
                            self.stones[move.x][move.y] = move.color
        else:
            raise StateError("Confirming a forbidden state")

    def put(self, move):
        """
        Check if the move passed as argument can be performed.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for execution.

        """

        x_ = move.x
        y_ = move.y
        color = move.color
        enem_color = enemy(color)

        if self.stones[x_][y_] == 'E':
            self.stones[x_][y_] = color
            self.lastdel = set()

            # check if kill (attack advantage)
            enemies = []
            for row, col in connected(x_, y_):
                neighcolor = self.stones[row][col]
                if neighcolor == enem_color:
                    enemies.append((row, col))
            for x, y in enemies:
                group, nblibs = self._data(x, y)
                if nblibs == 0:
                    for k, l in group:
                        self.lastdel.add(Move(enem_color, k, l))
                    self.last = move

                # check for suicide play if need be
            retval = True, self.lastdel
            if not self.last:
                _, nblibs = self._data(x_, y_)
                if not nblibs:
                    retval = False, "Suicide"
                    self.last = None
                else:
                    self.last = move

            # cancel
            self.stones[x_][y_] = 'E'
        else:
            retval = False, "Occupied"
            self.last = None

        return retval

    def remove(self, move):
        """
        Check if the move passed as argument can be undone. There is no notion of sequence.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for undo.

        """
        x_ = move.x
        y_ = move.y
        allowed = self.stones[x_][y_] == move.color
        if allowed:
            self.last = Move('E', x_, y_)
            data = self.deleted[-1]
            self.lastdel = None
        else:
            self.last = None
            data = "Empty" if self.stones[x_][y_] == 'E' else "Wrong Color."
        return allowed, data

    def _data(self, x, y, _group=None, _libs=None):
        """
        Returns the list of stones and the number of liberties of the group at (a, b).

        a, b -- the coordinates of any stone of the group.
        _group, _libs -- internal variables used in the recursion, no need to set them from outside.

        """
        color = self.stones[x][y]
        if _group is None:
            assert color != 'E'
            _group = []
            _libs = []
        if (x, y) not in _group:
            _group.append((x, y))
            for x, y in connected(x, y):
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


def connected(x, y):
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
        row = x + i
        if 0 <= row < gsize:
            col = y + j
            if 0 <= col < gsize:
                yield row, col


def enemy(color):
    if color == 'B':
        return 'W'
    elif color == 'W':
        return 'B'
    else:
        raise ValueError("No enemy for '{0}'".format(color))











