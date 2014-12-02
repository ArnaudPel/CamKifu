from tkinter import Button, Menu, StringVar, BooleanVar

from golib.gui.ui import UI, mod1


__author__ = 'Arnaud Peloquin'


class VUI(UI):
    """
    Extension of the GUI to add some vision-related commands.

    """

    # noinspection PyAttributeOutsideInit
    def init_components(self):
        UI.init_components(self)

        # Tkinter way to select buttons -- see add_bf() and add_sf()
        self.checkvar_onoff = BooleanVar()
        self.checkvar_onoff.set(True)
        self.radvar_bf = StringVar()
        self.radvar_sf = StringVar()

        m_detect = Menu(self.menubar)
        m_detect.add_checkbutton(label="Active", command=self.toggle_active, variable=self.checkvar_onoff)
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

        b_next = Button(self.buttons, text="Next", command=lambda: self.execute("next"))
        self.bind_all("<{0}-f>".format(mod1), lambda _: self.execute("next"))
        b_next.grid(row=5, column=0, columnspan=2)

        # b_debug = Button(self.buttons, text="Debug", command=lambda: self.checkvar_onoff.set(True))
        # b_debug.grid(row=5, column=0, columnspan=2)

    def toggle_active(self):
        self.execute("on" if self.checkvar_onoff.get() else "off")

    def add_bf(self, bf_class, callback):
        """
        Add the board finder to the menu.

        """
        label = bf_class.label
        self.m_board.add_radiobutton(label=label, command=lambda: callback(bf_class),
                                     variable=self.radvar_bf, value=label)

    def add_sf(self, sf_class, callback):
        """
        Add the stones finder to the menu.

        """
        try:
            label = sf_class.label
        except AttributeError:
            label = "None"
        self.m_stones.add_radiobutton(label=label, command=lambda: callback(sf_class),
                                      variable=self.radvar_sf, value=label)

    def select_bf(self, label):
        self.radvar_bf.set(label)
        if label is not None:
            self.checkvar_onoff.set(True)

    def select_sf(self, label):
        self.radvar_sf.set(label)
        if label is not None:
            self.checkvar_onoff.set(True)