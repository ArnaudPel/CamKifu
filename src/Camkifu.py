from Queue import Queue, Empty
from Tkinter import Tk
import platform
import os
import cv2

import golib_conf
# keep this line above other project imports to keep appname right
golib_conf.appname = "Camkifu"

from dev.vmanager_dev import VManagerSeq
from vgui.vui import VUI

import Golib
from core.vmanager import VManager
from core.controllerv import ControllerV, ControllerVSeq
from core.imgutil import show

__author__ = 'Kohistan'

"""
Application entry point.

"""


def main(video=0, nogui=False, sgf=None, bounds=(0, 1)):
    """
    gui --  Set to false to run the vision on main thread. Handy when needing to
            display images from inside loops during dev.
    video -- A file to use as video input.

    """
    if nogui:
        # run in dev mode, everything on the main thread
        vision = VManagerSeq(ControllerVSeq(kifufile=sgf), video=video, bounds=bounds)
        vision.run()

    else:
        root = Tk(className="Camkifu")
        app = VUI(root)
        app.pack()
        imqueue = Queue(maxsize=10)
        vthread = VManager(ControllerV(app, app, kifufile=sgf), imqueue=imqueue, video=video, bounds=bounds)

        def img_update():
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
            root.after(5, img_update)

        vthread.start()
        try:
            root.after(0, img_update)
            Golib.center(root)

            # mac OS special, to bring app to front at startup
            if "Darwin" in platform.system():
                os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')

            root.mainloop()
        finally:
            vthread.request_exit()


def get_argparser():
    parser = Golib.get_argparser()
    vhelp = "File to use as video feed. If absent, a live camera feed will is used."
    parser.add_argument("-v", "--video", help=vhelp, default=0)

    bhelp = "Video file bounds, expressed as ratios in [0, 1]. See openCV VideoCapture.set()"
    parser.add_argument("-b", "--bounds", default=(0, 1), help=bhelp, type=float, nargs=2, metavar="R")

    parser.add_argument("--nogui", help="Run without Tkinter GUI.", action="store_true")
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, nogui=args.nogui, sgf=args.sgf, bounds=args.bounds)