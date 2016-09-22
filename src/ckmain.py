import argparse
import queue
import tkinter

import cv2

from golib.config import golib_conf as gc
gc.appname = "Camkifu"  # keep this line above other project imports to have appname right
import glmain

import camkifu.core
from camkifu.core import imgutil
import camkifu.vgui


"""
Application entry point.

"""

def configure(win):
    xstart = gc.glocation[0] + win.winfo_reqwidth() + gc.rwidth * gc.gsize + 20
    xstart = min(xstart, gc.screenw - 100)
    from camkifu.config import cvconf
    cvconf.bf_loc = (xstart, 40)
    # stonesfinder's window is more likely to be smaller than boardfinder's
    cvconf.sf_loc = (xstart, int(gc.screenh * 3 / 5))


def img_update(imqueue):
    """
    Method to display opencv images on the GUI thread (otherwise it fails)

    """
    try:
        while True:
            name, img, vidproc, loc = imqueue.get_nowait()
            if img is not None:
                imgutil.show(img, name=name, loc=loc)
            else:
                imgutil.destroy_win(name)
    except queue.Empty:
        pass


def main(video=0, sgf=None, bounds=(0, 1), sf=None, bf=None, active=True):
    """
    singleth --  Set to True to run everything on the main thread. Handy when in need to
            display images from inside loops during dev.
    video -- Filename or device descriptor, as used in cv2.VideoCapture().

    """
    assert cv2.__version__ == "3.1.0"  # disable that if needs be, this is just meant as a quick indication
    root = tkinter.Tk(className="Camkifu")
    glmain.configure(root)
    app = camkifu.vgui.VUI(root, active=active)
    app.pack()
    configure(root)
    imqueue = queue.Queue(maxsize=10)
    controller = camkifu.vgui.ControllerV(app, app, sgffile=sgf, video=video, bounds=bounds)
    vmanager = camkifu.core.VManager(controller, imqueue=imqueue, bf=bf, sf=sf, active=active)

    def tk_routine():
        img_update(imqueue)
        root.after(5, tk_routine)

    root.after(0, tk_routine)
    glmain.place(root)
    glmain.bring_to_front()

    vmanager.start()
    root.mainloop()


def add_finder_args(parser: argparse.ArgumentParser):
    """
    Add board finder and stones finder arguments definition to the provided parser.

    """
    bfhelp = "Board finder class to instantiate at startup. Defaults to configuration defined in cvconf.py"
    parser.add_argument("--bf", help=bfhelp)

    sfhelp = "Stones finder class to instantiate at startup. Defaults to configuration defined in cvconf.py"
    parser.add_argument("--sf", help=sfhelp)


def get_argparser() -> argparse.ArgumentParser:
    """
    Get command line arguments parser. Is actually an enrichment of the Golib argument parser.

    """
    parser = glmain.get_argparser()
    vhelp = "Filename, or device, as used in cv2.VideoCapture(). Defaults to device \"0\"."
    parser.add_argument("-v", "--video", default=0, metavar="VID", help=vhelp)

    bhelp = "Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()"
    parser.add_argument("-b", "--bounds", default=(0, 1), type=float, nargs=2, metavar="R", help=bhelp)

    ahelp = "Don't activate the video processing at startup."
    parser.add_argument("--off", action='store_true', default=False, help=ahelp)

    add_finder_args(parser)

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, sgf=args.sgf, bounds=args.bounds, bf=args.bf, sf=args.sf, active=not args.off)
