from Queue import Queue, Empty
from Tkinter import Tk
import cv2
from cam.imgutil import show
from cam.video import VidRecorder, KeyboardInput

from cam.vthread import Vision
from config.devconf import vid_out_dir
from go.kifu import Kifu
from gui.controller import Controller
from gui.ui import UI

__author__ = 'Kohistan'


def main(gui=True):
    """
    gui --  Set to false to run the vision on main thread. Handy when needing to
            display images from inside loops during dev.

    """

    if gui:
        root = Tk()
        app = UI(root)
        control = Controller(Kifu.new(), app, app)

        imqueue = Queue(maxsize=10)
        vthread = Vision(control, imqueue)

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
        vthread = Vision(None, None)
        vthread.run()  # run on the main thread


def record():
    #noinspection PyArgumentList
    cam = cv2.VideoCapture(0)
    recorder = VidRecorder(cam, vid_out_dir, "Plaizac 1")

    kbin = KeyboardInput(recorder)
    kbin.daemon = True
    kbin.start()

    recorder.execute()


if __name__ == '__main__':
    main(gui=True)
    #record()
