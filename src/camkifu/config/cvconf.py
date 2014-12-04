from golib.config.golib_conf import gsize


__author__ = 'Arnaud Peloquin'

"""
Vision-related configuration.

"""

# the size in pixels of the canonical image (the rectangular image containing exactly the goban)
canonical_size = 25 * gsize

# shortest time (in seconds) between two processing: put thread to sleep if it's too early
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
    BoardFinderManual,
    BoardFinderAuto,
]

from camkifu.stone.sf_contours import SfContours
from camkifu.stone.sf_clustering import SfClustering
from camkifu.stone.sf_meta import SfMeta
from camkifu.stone.sf_tuto import StonesFinderTuto
from camkifu.stone.sf_sandbox import SfSandbox
from camkifu.stone.sf_bgsub2 import BackgroundSub2
from camkifu.stone.sf_bgsub import BackgroundSub
# the first element in the list will be loaded at startup.
sfinders = [
    # DummyFinder,
    SfMeta,
    SfClustering,
    SfSandbox,
    SfContours,
    None,
    StonesFinderTuto,
    BackgroundSub2,
    BackgroundSub,
]