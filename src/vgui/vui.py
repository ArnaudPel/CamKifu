from Tkinter import Button, Menu, StringVar
from gui.ui import UI

__author__ = 'Kohistan'


class VUI(UI):

    def init_components(self):
        UI.init_components(self)

        m_detect = Menu(self.menubar)
        self.m_board = Menu(m_detect)
        self.m_stones = Menu(m_detect)
        m_detect.add_cascade(label="Board", menu=self.m_board)
        m_detect.add_cascade(label="Stones", menu=self.m_stones)
        self.menubar.insert_cascade(0, label="Detection", menu=m_detect)

        b_pause = Button(self.buttons, text="Pause", command=lambda: self.execute("pause"))
        b_pause.grid(row=3, column=0)

        # annoying Tkinter way to select radiobuttons -- see add_bf() and add_sf()
        self.radvar_bf = StringVar()
        self.radvar_sf = StringVar()

    def add_bf(self, label, callback, select=False):
        self.m_board.add_radiobutton(label=label, command=callback, variable=self.radvar_bf, value=label)
        if select:
            self.radvar_bf.set(label)
            callback()

    def add_sf(self, label, callback, select=False):
        self.m_stones.add_radiobutton(label=label, command=callback, variable=self.radvar_sf, value=label)
        if select:
            self.radvar_sf.set(label)
            callback()