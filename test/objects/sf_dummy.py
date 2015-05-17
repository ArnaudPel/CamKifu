import golib.model
import camkifu.stone
from test import devconf


class DummyFinder(camkifu.stone.StonesFinder):
    """
    Can be used to simulate the detection of an arbitrary sequence of stones.
    Useful to test "test code". Double use of word 'test' intended :)

    """

    label = "Test SF"

    def __init__(self, vmanager):
        super().__init__(vmanager)
        self.ctype = devconf.dummy_sf_args[0]
        self.iterator = iter(devconf.dummy_sf_args[1])

    def _find(self, goban_img):
        try:
            mv_string = next(self.iterator)
            mv = golib.model.Move(self.ctype, string=mv_string)
            self.suggest(mv.color, mv.x, mv.y)
        except StopIteration:
            # self.interrupt()
            self.vmanager.stop_processing()

    def _learn(self):
        pass  # no user input expected as all move are already programmed.