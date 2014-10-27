from golib.config.golib_conf import gsize


__author__ = 'Arnaud Peloquin'

"""
Vision-related configuration.

"""

# the size in pixels of the canonical image (the rectangular image containing exactly the goban)
canonical_size = 25 * gsize

# width of the screen, pixels
screenw = 1920

# height of the screen, pixels
screenh = 1200

# shortest time (in seconds) between two processings: put thread to sleep if it's too early
frame_period = 0.2

# message to mark un-synchronization of threads reading a video file
unsynced = "unsynced"

# imports below must be done after variable declarations above,
# so that cyclic imports are avoided.

# from test.objects.sf_dummy import DummyFinder
from camkifu.board.bf_manual import BoardFinderManual
from camkifu.board.bf_auto import BoardFinderAuto
# the first element in the list will be loaded at startup.
bfinders = [
    BoardFinderAuto,
    BoardFinderManual,
]

from camkifu.stone.sf_bgsub import BackgroundSub
from camkifu.stone.sf_neighcomp import NeighbourComp
# the first element in the list will be loaded at startup.
sfinders = [
    # DummyFinder,
    BackgroundSub,
    NeighbourComp,
]


