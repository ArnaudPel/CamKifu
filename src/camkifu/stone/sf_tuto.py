from numpy import zeros_like
from camkifu.core.imgutil import draw_str
from camkifu.stone.stonesfinder import StonesFinder
from golib.config.golib_conf import gsize, B


class StonesFinderTuto(StonesFinder):
    """
    This class has been used to put together a tutorial on how to create a new StonesFinder. Methods can be renamed to
    quickly run a particular step of the tutorial.

    """

    label = "Stones Tuto"

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.canvas = None

    def _find(self, goban_img):
        # check emptiness to avoid complaints since this method will be called in a loop
        if self.is_empty(2, 12):
            # using "numpy" coordinates frame for x and y
            self.suggest(B, 2, 12)
        if self.is_empty(12, 2):
            # using "opencv" coordinates frame for x and y
            self.suggest(B, 2, 12, 'tk')

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
        draw_str(goban_img, "Hello stones finding tutorial !")
        self._show(goban_img)

    def _find_suggest(self, _):
        """
        Implemnentation 2 of _find() from the tutorial.

        """
        # check emptiness to avoid complaints since this method will be called in a loop
        if self.is_empty(2, 12):
            # using "numpy" coordinates frame for x and y
            self.suggest(B, 2, 12)
        if self.is_empty(12, 2):
            # using "opencv" coordinates frame for x and y
            self.suggest(B, 2, 12, 'tk')

    def _find_getrect(self, goban_img):
        """
        Implemnentation 3 of _find() from the tutorial.

        """
        canvas = zeros_like(goban_img)
        for r in range(gsize):      # row index
            for c in range(gsize):  # column index
                if r == c or r == gsize - c - 1:
                    x0, y0, x1, y1 = self._getrect(r, c)
                    canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
        self._show(canvas)

    def _find_border(self, goban_img):
        """
        Implemnentation 4 of _find() from the tutorial.

        """
        canvas = zeros_like(goban_img)
        for r, c in self._empties_border(2):  # 2 is the line height as in go vocabulary (0-based)
            x0, y0, x1, y1 = self._getrect(r, c)
            canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
        self._show(canvas)

    def _find_spiral(self, goban_img):
        """
        Implemnentation 5 of _find() from the tutorial.

        """
        count = 0
        if self.canvas is None:
            self.canvas = zeros_like(goban_img)
        for r, c in self._empties_spiral():
            if count == self.total_f_processed % gsize ** 2:
                x0, y0, x1, y1 = self._getrect(r, c)
                self.canvas[x0:x1, y0:y1] = goban_img[x0:x1, y0:y1]
                break
            count += 1
        self.last_shown = 0  # force display of all images
        self._show(self.canvas)