from Tkconstants import TOP
from Tkinter import Button
from gui.ui import UI

__author__ = 'Kohistan'


class VUI(UI):

    def init_components(self):
        UI.init_components(self)
        b_pause = Button(self.buttons, text="Pause", command=lambda: self.execute("pause"))
        b_pause.grid(row=3, column=0)
