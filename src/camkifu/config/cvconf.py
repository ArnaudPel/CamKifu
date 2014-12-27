from golib.config.golib_conf import gsize


__author__ = 'Arnaud Peloquin'

"""
Vision-related configuration.

"""

# the size in pixels of the canonical image (the rectangular image containing exactly the goban)
canonical_size = 20 * gsize

# shortest time (in seconds) between two processing: put thread to sleep if it's too early
frame_period = 0.2

# message to mark un-synchronization of threads reading a video file
unsynced = "unsynced"

# file read frames per second (can be used to skip frames automatically). eg 5 frames per sec is still fast for Go.
file_fps = 5

# imports below must be done after variable declarations above,
# so that cyclic imports are avoided.

# from test.objects.sf_dummy import DummyFinder
from camkifu.board.bf_manual import BoardFinderManual
from camkifu.board.bf_auto import BoardFinderAuto

# the first element in the list will be loaded at startup.
bfinders = [
    BoardFinderManual,
    BoardFinderAuto,
]

from camkifu.stone.sf_contours import SfContours
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_meta import SfMeta
from camkifu.stone.sf_tuto import StonesFinderTuto
from camkifu.stone.sf_sandbox import SfSandbox
# the first element in the list will be loaded at startup.
sfinders = [
    # DummyFinder,
    SfMeta,
    SfClustering,
    SfSandbox,
    SfContours,
    None,
    StonesFinderTuto,
]

# location of the board_finder window. set to None to center
bf_loc = None

# location of the stones_finder window. set to None to center
sf_loc = None
