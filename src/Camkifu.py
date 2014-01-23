from Queue import Queue, Empty
from Tkinter import Tk
import platform
import os
import cv2

import golib_conf
# keep this line above other project imports to keep appname right
golib_conf.appname = "Camkifu"

from vgui.vui import VUI

import Golib
from core.vmanager import VManager
from vgui.controllerv import ControllerV
from core.imgutil import show

__author__ = 'Arnaud Peloquin'

"""
Application entry point.

"""


def img_update(imqueue):
    # method to display opencv images on the GUI thread (otherwise it fails)
    try:
        while True:
            name, img, vidproc = imqueue.get_nowait()
            if img is not None:
                show(img, name=name)
                key = cv2.waitKey(20)
                vidproc.key = key
            else:
                cv2.destroyWindow(name)
    except Empty:
        pass


def main(video=0, sgf=None, bounds=(0, 1)):
    """
    singleth --  Set to True to run everything on the main thread. Handy when in need to
            display images from inside loops during dev.
    video -- Filename or device descriptor, as used in cv2.VideoCapture().

    """
    root = Tk(className="Camkifu")
    app = VUI(root)
    app.pack()
    imqueue = Queue(maxsize=10)
    controller = ControllerV(app, app, sgffile=sgf, video=video, bounds=bounds)
    vthread = VManager(controller, imqueue=imqueue)

    def tk_routine():
        img_update(imqueue)
        root.after(5, tk_routine)

    vthread.start()
    try:
        root.after(0, tk_routine)
        Golib.center(root)

        # mac OS special, to bring app to front at startup
        if "Darwin" in platform.system():
            os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

        root.mainloop()
    finally:
        vthread.request_exit()


def get_argparser():
    parser = Golib.get_argparser()
    vhelp = "Filename, or device, as used in cv2.VideoCapture(). Defaults to device \"0\"."
    parser.add_argument("-v", "--video", default=0, help=vhelp)

    bhelp = "Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()"
    parser.add_argument("-b", "--bounds", default=(0, 1), type=float, nargs=2, metavar="R", help=bhelp)

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, sgf=args.sgf, bounds=args.bounds)