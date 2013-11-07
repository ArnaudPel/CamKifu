from threading import Thread
import time
from gui.pipewarning import PipeWarning

__author__ = 'Kohistan'


class AutoClick(Thread):
    """ Thread dedicated to display """

    def __init__(self, goban, *args, **kwargs):
        super(AutoClick, self).__init__(*args, **kwargs)
        self.goban = goban

    def run(self, *args, **kwargs):
        i = 0
        try:
            while True:
                if i / 10 % 2 == 0:
                    event = "<Right>"
                else:
                    event = "<Left>"
                self.goban.pipe("event", event)
                i += 1
                time.sleep(1)
        except PipeWarning as pwa:
            print pwa


