import threading
import time

__author__ = 'Kohistan'


class Vision(threading.Thread):

    """ Thread dedicated to display """

    def __init__(self, goban, *args, **kwargs):
        super(Vision, self).__init__(*args, **kwargs)
        self.goban = goban

    def run(self, *args, **kwargs):
        i = 0
        while True:
            time.sleep(1)
            if i/10 % 2 == 0:
                event = "<Button-1>"
            else:
                event = "<Button-2>"
            self.goban.canvas.event_generate(event)
            self.goban.canvas.update()
            i += 1


