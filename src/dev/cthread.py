from Tkinter import Tk
from threading import Thread
from time import sleep
from core.warnings import PipeWarning
from go.kifu import Kifu
from gui.controller import Controller
from vgui.vui import VUI

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
                sleep(0.1)
        except PipeWarning as pwa:
            print pwa

if __name__ == '__main__':
    kifu = "/Users/Kohistan/Documents/go/Perso Games/MrYamamoto-Kohistan.sgf"
    root = Tk()
    app = VUI(root)
    control = Controller(app, app, kifu)

    cthread = AutoClick(control)
    cthread.start()

    root.mainloop()

