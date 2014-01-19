
from golib_conf import gsize


__author__ = 'Kohistan'

"""
Vision-related configuration.

"""

canonical_size = 25 * gsize

dummy_sf_args = ("kgs",
                ["W[H8]", "B[J8]", "W[K12]", "B[F12]", "W[F11]", "B[H10]",
                 "W[J14]", "B[J12]", "W[J11]", "B[J13]", "W[K13]"]
)


# imports below must be done after variable declarations above,
# so that cyclic imports are avoided

from board.board1 import BoardFinderManual
from board.board2 import BoardFinderAuto
# the first element in the list will be loaded at startup
bfinders = [
    BoardFinderManual,
    BoardFinderAuto,
]

from stone.stones1 import BackgroundSub
from stone.stones2 import NeighbourComp
from stone.stonesfinder import DummyFinder
# the first element in the list will be loaded at startup
sfinders = [
    BackgroundSub,
    DummyFinder,
    NeighbourComp,
]


