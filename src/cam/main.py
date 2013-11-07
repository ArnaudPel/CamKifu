from Queue import Queue, Empty
from Tkinter import Tk
import cv2
import sys
from cam.imgutil import show
from cam.video import VidSampler, KeyboardInput

from cam.vthread import Vision
from config.devconf import vid_out_dir
from go.kifu import Kifu
from gui.goban import Goban

__author__ = 'Kohistan'


def main():
    root = Tk()
    goban = Goban(root, Kifu())
    imqueue = Queue(maxsize=10)

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

    vthread = Vision(goban, imqueue)
    vthread.start()
    root.after(0, img_update)

    try:
        root.mainloop()
    finally:
        vthread.request_exit()


def record():
    cam = cv2.VideoCapture(0)
    recorder = VidSampler(cam, vid_out_dir, "Plaizac 1")

    kbin = KeyboardInput(recorder)
    kbin.daemon = True
    kbin.start()

    recorder.execute()


if __name__ == '__main__':
    #main()
    record()