from Queue import Queue, Empty
from Tkinter import Tk
from sys import argv
from dev.vmanager_dev import VManagerSeq
import cv2

from go.kifu import Kifu
from vgui.vui import VUI

from core.vmanager import VManager
from core.controllerv import ControllerV, ControllerVSeq
from core.imgutil import show

__author__ = 'Kohistan'

"""
Application entry point.

"""


def main(gui=True):
    """
    gui --  Set to false to run the vision on main thread. Handy when needing to
            display images from inside loops during dev.

    """

    if gui:
        root = Tk()
        app = VUI(root)
        app.pack()
        control = ControllerV(Kifu.new(), app, app)

        imqueue = Queue(maxsize=10)
        vthread = VManager(control, imqueue)

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
    else:
        # run in dev mode, everything on the main thread
        vision = VManagerSeq(ControllerVSeq(Kifu.new()))
        vision.run()


if __name__ == '__main__':

    if "nogui" in argv:
        main(gui=False)
    else:
        main()