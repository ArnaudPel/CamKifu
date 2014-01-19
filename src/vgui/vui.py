from Tkinter import Button, Menu, StringVar
from gui.ui import UI

__author__ = 'Kohistan'


class VUI(UI):

    # noinspection PyAttributeOutsideInit
    def init_components(self):
        UI.init_components(self)

        m_detect = Menu(self.menubar)
        self.m_board = Menu(m_detect)
        self.m_stones = Menu(m_detect)
        m_detect.add_cascade(label="Board", menu=self.m_board)
        m_detect.add_cascade(label="Stones", menu=self.m_stones)
        self.menubar.insert_cascade(0, label="Detection", menu=m_detect)

        m_video = Menu(self.menubar)
        m_video.add_command(label="Video File...", command=lambda: self.execute("vidfile"))
        m_video.add_command(label="Live video", command=lambda: self.execute("vidlive"))
        self.menubar.insert_cascade(0, label="Video", menu=m_video)

        b_run = Button(self.buttons, text="Run", command=lambda: self.execute("run"))
        b_pause = Button(self.buttons, text="Pause", command=lambda: self.execute("pause"))
        b_run.grid(row=4, column=0)
        b_pause.grid(row=4, column=1)

        # annoying Tkinter way to select radiobuttons -- see add_bf() and add_sf()
        self.radvar_bf = StringVar()
        self.radvar_sf = StringVar()

    def add_bf(self, bf_class, callback, select=False):
        label = bf_class.label
        self.m_board.add_radiobutton(label=label, command=lambda: callback(bf_class),
                                     variable=self.radvar_bf, value=label)
        if select:
            self.radvar_bf.set(label)
            callback(bf_class)

    def add_sf(self, sf_class, callback, select=False):
        label = sf_class.label
        self.m_stones.add_radiobutton(label=label, command=lambda: callback(sf_class),
                                      variable=self.radvar_sf, value=label)
        if select:
            self.radvar_sf.set(label)
            callback(sf_class)