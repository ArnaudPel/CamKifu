from threading import RLock
from config.guiconf import gsize
from go.sgf import Move
from go.stateerror import StateError


__author__ = 'Kohistan'

"""
Hold the current state of a game, and ensure logical consistency of changes
made to that state.

"""

import numpy as np


class RuleUnsafe(object):
    """
    This class is not thread safe.
    Its consistency is highly dependent on the good usage of self.confirm()

    """

    def __init__(self):
        self.stones = np.empty((gsize, gsize), dtype=np.str)
        self.stones.fill('E')
        self.deleted = []

        self.stones_buff = self.stones.copy()
        self.deleted_buff = []

    def confirm(self):
        """
        Persist the state of the last check, either next() or previous()

        """
        if self.stones_buff is not None:
            self.stones = self.stones_buff
            self.deleted = self.deleted_buff
        else:
            raise StateError("Confirmation Denied")

    def reset(self):
        """ Clear any unconfirmed data. """
        self.stones_buff = self.stones.copy()
        self.deleted_buff = list(self.deleted)

    def put(self, move, reset=True):
        """
        Check if the move passed as argument can be performed.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for execution.

        """
        if reset:
            self.reset()
        x_ = move.x
        y_ = move.y
        color = move.color
        enem_color = enemy(color)

        if self.stones_buff[x_][y_] == 'E':
            self.stones_buff[x_][y_] = color

            # check if kill (attack advantage)
            enemies = []
            deleted = set()
            safe = False
            self.deleted_buff.append(deleted)
            for row, col in connected(x_, y_):
                neighcolor = self.stones_buff[row][col]
                if neighcolor == enem_color:
                    enemies.append((row, col))
            for x, y in enemies:
                group, nblibs = self._data(x, y)
                if nblibs == 0:
                    safe = True  # killed at least one enemy
                    for k, l in group:
                        deleted.add(Move(enem_color, k, l))

            # check for suicide play if need be
            retval = True, deleted
            if not safe:
                _, nblibs = self._data(x_, y_)
                if not nblibs:
                    retval = False, "Suicide"
        else:
            retval = False, "Occupied"

        return retval

    def remove(self, move, reset=True):
        """
        Check if the move passed as argument can be undone. There is no notion of sequence.

        Note that the state of this rule object will not be updated after this call,
        meaning that from its point of vue the move has not happened.
        To update the state, please confirm()

        move -- the move to check for undo.

        """
        if reset:
            self.reset()
        x_ = move.x
        y_ = move.y

        allowed = self.stones_buff[x_][y_] == move.color
        if allowed:
            data = self.deleted_buff.pop()
            self.stones_buff[x_][y_] = 'E'
        else:
            data = "Empty" if self.stones_buff[x_][y_] == 'E' else "Wrong Color."
        return allowed, data

    def _data(self, x, y, _group=None, _libs=None):
        """
        Returns the list of stones and the number of liberties of the group at (a, b).

        a, b -- the coordinates of any stone of the group.
        _group, _libs -- internal variables used in the recursion, no need to set them from outside.

        """
        color = self.stones_buff[x][y]
        if _group is None:
            assert color != 'E'
            _group = []
            _libs = []
        if (x, y) not in _group:
            _group.append((x, y))
            for x, y in connected(x, y):
                neighcolor = self.stones_buff[x][y]
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


class Rule(RuleUnsafe):

    def put(self, move, reset=True):
        with RLock():
            return super(Rule, self).put(move, reset)

    def remove(self, move, reset=True):
        with RLock():
            return super(Rule, self).remove(move, reset)

    def confirm(self):
        """
        Top level needs must also acquire a lock to wrap operation (e.g. put or remove) and confirmation.
        Otherwise another thread can perform an operation, and reset the first operation.

        """
        with RLock():
            return super(Rule, self).confirm()


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











