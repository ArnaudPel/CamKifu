from argparse import ArgumentParser
from os.path import basename
from queue import Queue
from tkinter import Tk

from camkifu.core.vmanager import VManager
import glmain, ckmain
from test.objects.kifuref import display_matcher, print_matcher
from test.objects.controllerv_test import ControllerVTest


__author__ = 'Arnaud Peloquin'

"""
Test the default configuration of BoardFinder / StonesFinder on a recorded game (video file), and compare
moves found with an associated reference SGF. This script is useful to test a global configuration, as errors
can originate from board or stones algorithms indifferently.

"""


def set_title(root2, video, vmanager):
    """
    Set the title of the result window.

    """
    src = "Live video"
    if type(video) is str:
        src = basename(video)
    title = "{} [{}, {}]".format(src, vmanager.bf_class.__name__, vmanager.sf_class.__name__)
    root2.title(title)


def main(ref_sgf, video=0, vid_bounds=(0, 1), mv_bounds=0, failfast=False, bf=None, sf=None):
    root = Tk(className="Detection Test")
    root.withdraw()
    imqueue = Queue(maxsize=10)
    controller = ControllerVTest(ref_sgf, video=video, vid_bounds=vid_bounds, mv_bounds=mv_bounds, failfast=failfast)
    vmanager = VManager(controller, imqueue=imqueue, bf=bf, sf=sf)
    vmanager.start()

    def tk_routine():
        ckmain.img_update(imqueue)
        if vmanager.hasrun and not vmanager.is_processing():
            root.destroy()
        else:
            root.after(5, tk_routine)

    try:
        root.after(5, tk_routine)
        glmain.bring_to_front()
        root.mainloop()
    finally:
        vmanager.stop_processing()

    # display test results
    root2 = Tk()
    set_title(root2, video, vmanager)
    matcher = controller.kifu.check()
    frame = display_matcher(matcher, master=root2)
    print_matcher(matcher)
    frame.pack()
    glmain.center(root2)
    glmain.bring_to_front()
    root2.mainloop()


def get_argparser() -> ArgumentParser:
    parser = ckmain.get_argparser()

    # compulsory argument
    shelp = "SGF file to use as reference during test. Required."
    parser.add_argument("--sgf", required=True, help=shelp)

    # optional arguments
    parser.add_argument("--failfast", help="Fail and stop test at first wrong move.", action="store_true")

    mhelp = "The subsequence of moves to consider in the reference sgf." \
            "Provide first and last move number of interest (1-based numeration)." \
            "Previous moves will be used to initialize the \"working\" sgf."
    parser.add_argument("-m", "--moves", default=(0, 1000), type=int, nargs=2, metavar="M", help=mhelp)
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args.sgf, video=args.video, vid_bounds=args.bounds, mv_bounds=args.moves, failfast=args.failfast,
         bf=args.bf, sf=args.sf)
