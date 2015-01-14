import time

import numpy
from golib.config.golib_conf import gsize, B, W, E

import camkifu.stone
from camkifu.core import imgutil


class StonesFinderTuto(camkifu.stone.StonesFinder):
    """
    This class has been used to put together a tutorial on how to create a new StonesFinder. Methods can be renamed to
    quickly run a particular step of the tutorial.

    """

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.canvas = None

    def _find(self, goban_img):
        # using "numpy" coordinates frame for x and y
        black = ((W, 8, 8), (W, 8, 10), (W, 10, 8), (W, 10, 10))
        white = ((B, 7, 7), (B, 7, 11), (B, 11, 7), (B, 11, 11), (B, 9, 9))
        add = black if self.total_f_processed % 2 else white
        rem = white if self.total_f_processed % 2 else black
        moves = []
        for color, r, c in add:
            moves.append((color, r, c))
        for _, r, c in rem:
            if not self.is_empty(r, c):
                moves.append((E, r, c))
        time.sleep(0.7)
        self.bulk_update(moves)

    def _learn(self):
        pass

    # ------------------------------------------------------
    #
    #               TUTORIAL STEPS
    #
    # ------------------------------------------------------

    def _find_minimal(self, goban_img):
        """
        Implemnentation 1 of _find() from the tutorial.

        """
        imgutil.draw_str(goban_img, "Hello stones finding tutorial !")
        self._show(goban_img)

    def _find_suggest(self, _):
        """
        Implemnentation 2 of _find() from the tutorial.

        """
        # check emptiness to avoid complaints since this method will be called in a loop
        if self.is_empty(2, 12):
            # using "numpy" coordinates frame for x and y
            self.suggest(B, 2, 12)

    def _find_bulk(self, _):
        # using "numpy" coordinates frame for x and y
        black = ((W, 8, 8), (W, 8, 10), (W, 10, 8), (W, 10, 10))
        white = ((B, 7, 7), (B, 7, 11), (B, 11, 7), (B, 11, 11), (B, 9, 9))
        add = black if self.total_f_processed % 2 else white
        rem = white if self.total_f_processed % 2 else black
        moves = []
        for color, r, c in add:
            moves.append((color, r, c))
        for _, r, c in rem:
            if not self.is_empty(r, c):
                moves.append((E, r, c))
        time.sleep(0.7)
        self.bulk_update(moves)

    def _find_getrect(self, goban_img):
        """
        Implemnentation 3 of _find() from the tutorial.

        """
        canvas = numpy.zeros_like(goban_img)
        for r in range(gsize):      # row index
            for c in range(gsize):  # column index
                if r == c or r == gsize - c - 1:
                    x0, y0, x1, y1 = self.getrect(r, c)
                    canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
        self._show(canvas)

    def _find_border(self, goban_img):
        """
        Implemnentation 4 of _find() from the tutorial.

        """
        canvas = numpy.zeros_like(goban_img)
        for r, c in self._empties_border(2):  # 2 is the line height as in go vocabulary (0-based)
            x0, y0, x1, y1 = self.getrect(r, c)
            canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
        self._show(canvas)

    def _find_spiral(self, goban_img):
        """
        Implemnentation 5 of _find() from the tutorial.

        """
        count = 0
        if self.canvas is None:
            self.canvas = numpy.zeros_like(goban_img)
        for r, c in self._empties_spiral():
            if count == self.total_f_processed % gsize ** 2:
                x0, y0, x1, y1 = self.getrect(r, c)
                self.canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
                break
            count += 1
        self.last_shown = 0  # force display of all images
        self._show(self.canvas)