from numpy import ndarray, array
from golib.gui.controller import ControllerBase

__author__ = 'Arnaud Peloquin'


class ControllerVDev(ControllerBase):
    """
    A no-GUI base controller for "test" code.

    """

    def __init__(self, sgffile=None, video=0, bounds=(0, 1)):
        super().__init__(sgffile=sgffile)
        self.video = video
        self.bounds = bounds
        self.api = {
            "append": self.cvappend,
            "delete": self._delete,
            "bulk": self._bulk_update,
        }

    def pipe(self, instruction, *args):
        """
        Execute instruction straight away on the caller thread.

        """
        self.api[instruction](*args)

    def cvappend(self, move):
        """
        Put the move in the current Rule object, then append it to the kifu.

        """
        move.number = self.current_mn + 1
        self.rules.put(move)
        self._append(move)

    def get_stones(self) -> ndarray:
        """
        Return a copy of the current goban state, in the numpy coordinates system.

        """
        return array(self.rules.stones, dtype=object).T
