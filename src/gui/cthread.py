from Tkinter import Tk
from threading import Thread
import time
from go.kifu import Kifu
from gui.controller import Controller
from gui.pipewarning import PipeWarning
from gui.ui import UI

__author__ = 'Kohistan'


class AutoClick(Thread):
    """
    An example of controlling the GUI programmatically.
    """

    def __init__(self, controller, *args, **kwargs):
        super(AutoClick, self).__init__(*args, **kwargs)
        self.ctrl = controller

    def run(self, *args, **kwargs):
        i = 0
        try:
            while True:
                if i / 100 % 2 == 0:
                    event = "<Right>"
                else:
                    event = "<Left>"
                self.ctrl.pipe("event", event)
                i += 1
                time.sleep(0.1)
        except PipeWarning as pwa:
            print pwa

if __name__ == '__main__':
    root = Tk()
    kifu = Kifu.parse("/Users/Kohistan/Documents/go/Perso Games/MrYamamoto-Kohistan.sgf")
    app = UI(root)
    control = Controller(kifu, app, app)

    cthread = AutoClick(control)
    cthread.start()

    root.mainloop()

