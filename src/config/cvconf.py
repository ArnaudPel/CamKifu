from golib_conf import gsize


__author__ = 'Kohistan'

"""
Vision-related configuration.

"""

# the size in pixels of the canonical image (the rectangular image containing exactly the goban)
canonical_size = 25 * gsize


# imports below must be done after variable declarations above,
# so that cyclic imports are avoided.

from board.bf_manual import BoardFinderManual
from board.bf_auto import BoardFinderAuto
# the first element in the list will be loaded at startup.
bfinders = [
    BoardFinderManual,
    BoardFinderAuto,
]

from stone.sf_bgsub import BackgroundSub
from stone.sf_neighcomp import NeighbourComp
# the first element in the list will be loaded at startup.
sfinders = [
    BackgroundSub,
    NeighbourComp,
]


