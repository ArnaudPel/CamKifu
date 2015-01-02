from argparse import ArgumentParser
from queue import Queue, Empty
from tkinter import Tk

import cv2


# keep this line above other project imports to keep appname right
from golib.config import golib_conf as gc

gc.appname = "Camkifu"

from camkifu.vgui.vui import VUI

import glmain
from camkifu.core.vmanager import VManager
from camkifu.vgui.controllerv import ControllerV
from camkifu.core.imgutil import show, destroy_win

__author__ = 'Arnaud Peloquin'

"""
Application entry point.

"""


def configure(win):
    xstart = gc.glocation[0] + win.winfo_reqwidth() + gc.rwidth * gc.gsize + 20
    xstart = min(xstart, gc.screenw - 100)
    import camkifu.config.cvconf as cvc
    cvc.bf_loc = (xstart, 40)
    # stonesfinder's window is more likely to be smaller than boardfinder's
    cvc.sf_loc = (xstart, int(gc.screenh * 3 / 5))


def img_update(imqueue):
    """
    Method to display opencv images on the GUI thread (otherwise it fails)

    """
    try:
        while True:
            name, img, vidproc, loc = imqueue.get_nowait()
            if img is not None:
                show(img, name=name, loc=loc)
                key = cv2.waitKey(20)
                vidproc.key = key
            else:
                destroy_win(name)
    except Empty:
        pass


def main(video=0, sgf=None, bounds=(0, 1), sf=None, bf=None):
    """
    singleth --  Set to True to run everything on the main thread. Handy when in need to
            display images from inside loops during dev.
    video -- Filename or device descriptor, as used in cv2.VideoCapture().

    """
    assert cv2.__version__ == "3.0.0-beta"  # disable that if needs be, this is just meant as a quick indication
    root = Tk(className="Camkifu")
    glmain.configure(root)
    app = VUI(root)
    app.pack()
    configure(root)
    imqueue = Queue(maxsize=10)
    controller = ControllerV(app, app, sgffile=sgf, video=video, bounds=bounds)
    vmanager = VManager(controller, imqueue=imqueue, bf=bf, sf=sf)

    def tk_routine():
        img_update(imqueue)
        root.after(5, tk_routine)

    root.after(0, tk_routine)
    glmain.place(root)
    glmain.bring_to_front()

    vmanager.start()
    root.mainloop()


def add_finder_args(parser: ArgumentParser):
    """
    Add board finder and stones finder arguments definition to the provided parser.

    """
    bfhelp = "Board finder class to instantiate at startup. Defaults to configuration defined in cvconf.py"
    parser.add_argument("--bf", help=bfhelp)

    sfhelp = "Stones finder class to instantiate at startup. Defaults to configuration defined in cvconf.py"
    parser.add_argument("--sf", help=sfhelp)


def get_argparser() -> ArgumentParser:
    """
    Get command line arguments parser. Is actually an enrichment of the Golib argument parser.

    """
    parser = glmain.get_argparser()
    vhelp = "Filename, or device, as used in cv2.VideoCapture(). Defaults to device \"0\"."
    parser.add_argument("-v", "--video", default=0, metavar="VID", help=vhelp)

    bhelp = "Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()"
    parser.add_argument("-b", "--bounds", default=(0, 1), type=float, nargs=2, metavar="R", help=bhelp)

    add_finder_args(parser)

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, sgf=args.sgf, bounds=args.bounds, bf=args.bf, sf=args.sf)