from Queue import Queue, Empty
from Tkinter import Tk
import cv2
from cam.imgutil import show

from cam.vthread import Vision
from go.kifu import Kifu
from gui.goban import Goban

__author__ = 'Kohistan'


def main():
    root = Tk()
    goban = Goban(root, Kifu())
    imqueue = Queue(maxsize=10)

    vthread = Vision(goban, imqueue)
    vthread.start()

    def img_update():
        try:
            while True:
                elem = imqueue.get_nowait()
                if elem is None:
                    cv2.destroyAllWindows()
                else:
                    name, img, vidproc = elem
                    show(img, name=name)
                    key = cv2.waitKey(20)
                    vidproc.key = key
        except Empty:
            pass
        root.after(5, img_update)

    root.after(0, img_update)

    try:
        root.mainloop()
    finally:
        vthread.request_exit()

if __name__ == '__main__':
    main()