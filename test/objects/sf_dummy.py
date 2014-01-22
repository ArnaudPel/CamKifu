from go.move import Move
from stone.stonesfinder import StonesFinder
from test.devconf import dummy_sf_args

__author__ = 'Kohistan'


class DummyFinder(StonesFinder):
    """
    Can be used to simulate the detection of an arbitrary sequence of stones.
    Useful to test "test code". Double use of word 'test' intended :)

    """

    label = "Test SF"

    def __init__(self, vmanager):
        super(DummyFinder, self).__init__(vmanager)
        self.ctype = dummy_sf_args[0]
        self.iterator = iter(dummy_sf_args[1])

    def _find(self, goban_img):
        try:
            move = self.iterator.next()
            self.suggest(Move(self.ctype, string=move))
        except StopIteration:
            self.interrupt()

    def _learn(self):
        pass  # no user input expected as all move are already programmed.