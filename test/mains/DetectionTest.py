from queue import Queue
from tkinter import Tk
import os
import platform
from threading import Thread
from time import sleep

from camkifu.core.vmanager import VManager
import CkMain
from test.objects.kifuref import display_matcher, print_matcher
from test.objects.controllerv_test import ControllerVTest


__author__ = 'Arnaud Peloquin'

"""
Test the default configuration of BoardFinder / StonesFinder on a recorded game (video file), and compare
moves found with an associated reference SGF. This script is useful to test a global configuration, as errors
can originate from board or stones algorithms indifferently.

"""


class ImgUpdater(Thread):

    def __init__(self, imqueue, parent_thread):
        super(ImgUpdater, self).__init__()
        self.daemon = True
        self.imqueue = imqueue
        self.parentth = parent_thread

    def run(self):
        while self.parentth.isAlive():
            CkMain.img_update(self.imqueue)
            sleep(0.1)


def main(reffile, sgffile=None, move_nr=0, failfast=False, bounds=(0, 1), video=0):
    root = Tk(className="Detection Test")
    root.withdraw()
    imqueue = Queue(maxsize=10)
    controller = ControllerVTest(reffile, sgffile=sgffile, video=video, vid_bounds=bounds,
                                 failfast=failfast, move_bounds=move_nr)
    vmanager = VManager(controller, imqueue=imqueue)
    vmanager.start()

    def tk_routine():
        if not vmanager.is_alive():
            root.destroy()
        else:
            CkMain.img_update(imqueue)
            root.after(5, tk_routine)

    try:
        root.after(0, tk_routine)
        # mac OS special, to bring app to front at startup
        if "Darwin" in platform.system():
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

        root.mainloop()
    finally:
        vmanager.stop_processing()

    # display test results
    root2 = Tk()
    matcher = controller.kifu.check()
    frame = display_matcher(matcher, master=root2)
    print_matcher(matcher)
    frame.pack()
    root2.mainloop()


def get_argparser():
    parser = CkMain.get_argparser()

    # compulsory argument
    parser.add_argument("sgf_ref", help="The SGF file to use as reference during test.")

    # optional arguments
    parser.add_argument("--failfast", help="Fail and stop test at first wrong move.", action="store_true")

    mhelp = "The subsequence of moves to consider in the reference sgf." \
            "Provide first and last move number of interest (1-based numeration)." \
            "Previous moves will be used to initialize the \"working\" sgf."
    parser.add_argument("-m", "--move", default=(0, 1000), type=int, nargs=2, metavar="M", help=mhelp)
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args.sgf_ref, sgffile=args.sgf, move_nr=args.move, failfast=args.failfast,
         bounds=args.bounds, video=args.video)
