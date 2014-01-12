from Queue import Queue
from Tkinter import Tk
from threading import Thread
from time import sleep
import Camkifu
from core.vmanager import VManager
from test.kifuref import display_matcher, print_matcher
from test.controllerv_test import ControllerVTest

__author__ = 'Kohistan'


class ImgUpdater(Thread):

    def __init__(self, imqueue, parent_thread):
        super(ImgUpdater, self).__init__()
        self.daemon = True
        self.imqueue = imqueue
        self.parentth = parent_thread

    def run(self):
        while self.parentth.isAlive():
            Camkifu.img_update(self.imqueue)
            sleep(0.1)


def main(reffile, sgffile=None, move_nr=0, failfast=False, bounds=(0, 1), video=0):
    controller = ControllerVTest(reffile, sgffile=sgffile, video=video, vid_bounds=bounds,
                                 failfast=failfast, move_bounds=move_nr)
    root = Tk()  # had to move that before ImgUpdater to avoid Python interpreter crash on my config.
    imqueue = Queue(10)
    vmanager = VManager(controller, imqueue=imqueue)
    vmanager.start()
    ImgUpdater(imqueue, vmanager).run()
    matcher = controller.kifu.check()

    frame = display_matcher(matcher, master=root)
    print_matcher(matcher)
    frame.pack()
    root.mainloop()


def get_argparser():
    parser = Camkifu.get_argparser()

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
