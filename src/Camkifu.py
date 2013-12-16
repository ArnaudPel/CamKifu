from Queue import Queue, Empty
from Tkinter import Tk
import Golib
from sys import argv
import cv2

import golib_conf
golib_conf.appname = "Camkifu"
from dev.vmanager_dev import VManagerSeq
from vgui.vui import VUI

from core.vmanager import VManager
from core.controllerv import ControllerV, ControllerVSeq
from core.imgutil import show

__author__ = 'Kohistan'

"""
Application entry point.

"""


def main(video=0, nogui=False, sgf=None):
    """
    gui --  Set to false to run the vision on main thread. Handy when needing to
            display images from inside loops during dev.
    video -- A file to use as video input.

    """
    if nogui:
        # run in dev mode, everything on the main thread
        vision = VManagerSeq(ControllerVSeq(kifufile=sgf), video=video)
        vision.run()

    else:
        root = Tk()
        app = VUI(root)
        app.pack()
        imqueue = Queue(maxsize=10)
        vthread = VManager(ControllerV(app, app, kifufile=sgf), imqueue, video=video)

        def img_update():
            try:
                while True:
                    elem = imqueue.get_nowait()
                    name, img, vidproc = elem
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
            root.mainloop()
        finally:
            vthread.request_exit()


def get_argparser():
    parser = Golib.get_argparser()
    vhelp = "File to use as video feed. If absent, a live camera feed will is used."
    parser.add_argument("-v", "--video", help=vhelp, default=0)
    parser.add_argument("--nogui", help="Run without tkinter GUI.", action="store_true")
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(video=args.video, nogui=args.nogui, sgf=args.sgf)