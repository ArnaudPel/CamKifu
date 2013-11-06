from threading import Thread
import time

__author__ = 'Kohistan'


class AutoClick(Thread):

    """ Thread dedicated to display """

    def __init__(self, goban, *args, **kwargs):
        super(AutoClick, self).__init__(*args, **kwargs)
        self.goban = goban

    def run(self, *args, **kwargs):
        i = 0
        while True:
            if self.goban.closed:
                break
            if i/10 % 2 == 0:
                event = "<Right>"
            else:
                event = "<Left>"
            self.goban.pipe(event)
            i += 1
            time.sleep(1)


