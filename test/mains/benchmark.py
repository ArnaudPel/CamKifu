import argparse
import os
import ntpath
import time

import ckmain
import camkifu.core
import test.objects


"""
Script that helps with running successive detection tests in one go. To be called on a directory containing
video files and reference sgf files.

For each sgf reference (supposed to contain the expected results), look for a file with the same name but a
different extension, and run the vision algorithms on that file. Stop when all references have been processed.

Expand / modify the Report class depending on the kind of information needed.

"""


def list_bench(refdir, vid_dir) -> dict:
    """
    Map each sgf reference to the first matching "video" file (if found) in a dict.
    The "video" file is the first file with a matching name, the extension is not checked.

    refdir -- the directory containing the sgf files
    vid_dir -- the directory containing the video files

    """
    # step 0: arrange paths
    if not vid_dir.endswith("/"):
        vid_dir += "/"
    if refdir is None:
        refdir = vid_dir
    elif not refdir.endswith("/"):
        refdir += "/"

    # step 1: list video candidates (all files that are not sgf)
    vid_files = []
    for f in os.listdir(vid_dir):
        if not f.endswith('.sgf'):
            vid_files.append(f)

    # step 2: match sgf references to video candidates
    bench = {}
    unmatched = []
    for sgf in os.listdir(refdir):
        if sgf.endswith('.sgf'):
            matched = False
            key = ntpath.basename(sgf)[:-4]
            for vf in vid_files:
                if ntpath.basename(vf).startswith(key):
                    matched = True
                    if sgf not in bench:
                        bench[refdir + sgf] = vid_dir + vf
                    else:
                        msg_base = "Benchmark warning: multiple video candidates for reference file {}.sgf : {}"
                        print(msg_base.format(key, [ntpath.basename(bench[sgf]), vf]))
                        break  # at least one warning should be enough to get attention
            if not matched:
                unmatched.append(ntpath.basename(sgf))
    if len(unmatched):
        print("Benchmark warning: unmatched reference files {}".format(unmatched))
    return bench


def main(vid_dir, refdir=None, bf=None, sf=None):
    """
    vid_dir -- the directory containing the videos on which to run the benchmark. May also contain the reference sgf(s).

    """
    bench = list_bench(refdir, vid_dir)
    reports = []
    for i, (sgf, vid) in enumerate(bench.items()):
        msg_base = "{} ({}/{}) ----------------------------------------------------------------"
        print(msg_base.format(ntpath.basename(vid), i+1, len(bench)))
        controller = test.objects.ControllerVTest(sgf, video=vid)
        vmanager = camkifu.core.VManager(controller, imqueue=DummyQueue(), bf=bf, sf=sf)
        stop_condition = lambda: vmanager.hasrun and not vmanager.is_processing()
        test.objects.ProcessKiller(vmanager, stop_condition).start()
        start = time.time()
        vmanager.run()
        reports.append(Report(vmanager, time.time() - start))
    for r in reports:
        print(r)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ckmain.add_finder_args(parser)

    # compulsory argument
    parser.add_argument("dir", help="Directory containing the videos on which to run the benchmark.")

    # optional arguments
    refdir_help = "Directory containing the reference sgfs associated with each video (matched on filename)." \
                  " Defaults to \"dir\" if omitted."
    parser.add_argument("--refdir", help=refdir_help)

    return parser


class Report():

    def __init__(self, vmanager, duration):
        self.vmanager = vmanager
        self.sgf = vmanager.controller.kifu.ref.sgffile
        self.video = vmanager.controller.video
        self.matcher = vmanager.controller.kifu.check()
        self.duration = duration

    def __repr__(self):
        name = ntpath.basename(self.sgf)[:-4]
        percent = round(100*self.matcher.ratio(), 1)
        return "[{}: {}% in {} s]".format(name, percent, int(round(self.duration)))


class DummyQueue():
    def put(self, x):
        pass

    def put_nowait(self, x):
        pass


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args.dir, refdir=args.refdir, bf=args.bf, sf=args.sf)

