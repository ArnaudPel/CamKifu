from os.path import realpath, join

from golib.config import golib_conf
"""
Vision-related configuration.

"""

# the size in pixels of the canonical image (the rectangular image containing exactly the goban)
canonical_size = 20 * golib_conf.gsize

# shortest time (in seconds) between two processing: put thread to sleep if it's too early
frame_period = 0.2

# message to mark un-synchronization of threads reading a video file
unsynced = "unsynced"

# file read frames per second (can be used to skip frames automatically). eg 5 frames per sec is still fast for Go.
file_fps = 5

# the first element in the list will be loaded at startup (unless specified otherwise in startup arguments)
# format: (module, class)
bfinders = [
    ("camkifu.board.bf_manual", "BoardFinderManual"),
    ("camkifu.board.bf_auto", "BoardFinderAuto"),
    ("None", "None"),
]

# the first element in the list will be loaded at startup (unless specified otherwise in startup arguments)
# format: (module, class)
sfinders = [
    ("camkifu.stone.sf_sandbox", "SfSandbox"),
    ("camkifu.stone.sf_neural", "SfNeural"),
    ("camkifu.stone.sf_meta", "SfMeta"),
    ("camkifu.stone.sf_contours", "SfContours"),
    ("camkifu.stone.sf_clustering", "SfClustering"),
    ("camkifu.stone.sf_tuto", "StonesFinderTuto"),
    ("None", "None"),
    # ("test.objects.sf_dummy", "DummyFinder"),
]

# location of the board_finder window. set to None to center
bf_loc = None

# location of the stones_finder window. set to None to center
sf_loc = None

fpath = realpath(__file__)
tmp_dir = fpath[:fpath.index('/src/')] + "/res/temp/"
gobanloc_npz = join(tmp_dir, "gobanlocs/")
tmp_sgf = join(tmp_dir, "auto-saved.sgf")

# TODO prompt user for that
# folder where to save the snapshots
train_dir = "/Users/Kohistan/Developer/PycharmProjects/Train"
snapshot_dir = "%s/data" % train_dir

sf_neural_model_url = "https://github.com/ArnaudPel/CamKifu/files/530135/sf-neural.model.30-09-2016.zip"
