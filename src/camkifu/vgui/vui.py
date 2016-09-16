import tkinter as tk
import golib.gui

from camkifu.config.cvconf import snapshot_dir
from camkifu.stone.training_ui import DataGeneration


# noinspection PyAttributeOutsideInit
class VUI(golib.gui.UI):
    """
    Extension of the GUI to add some vision-related commands.

    """

    def init_components(self):
        super().init_components()
        self.build_menu_training()
        self.build_menu_detection()
        self.build_menu_video()
        self.build_buttons()

    def build_buttons(self):
        self.video_pos = tk.Scale(self.buttons, command=lambda x: self.execute("vidpos", float(x)), orient=tk.HORIZONTAL)
        b_run = tk.Button(self.buttons, text="Run", command=lambda: self.execute("run"))
        b_pause = tk.Button(self.buttons, text="Pause", command=lambda: self.execute("pause"))
        b_run.grid(row=4, column=0)
        b_pause.grid(row=4, column=1)

        b_next = tk.Button(self.buttons, text="Next", command=lambda: self.execute("next"))
        self.bind_all("<{0}-f>".format(golib.gui.ui.mod1), lambda _: self.execute("next"))
        b_next.grid(row=5, column=0, columnspan=2)

        video_pos_lbl = tk.Label(self.buttons, text="Video position (%):")
        video_pos_lbl.grid(row=6, column=0, columnspan=2)
        self.video_pos.grid(row=7, column=0, columnspan=2)

        self.save_goban = tk.IntVar()  # whether to save the current game record with the snapshot (for ANN training)
        self.save_goban.set(1)

        def snap_cmd():
            self.execute("snapshot", self.save_goban.get())
        b_snap = tk.Button(self.buttons, text="Snapshot", command=snap_cmd)
        self.bind_all("<{0}-x>".format(golib.gui.ui.mod1), lambda _: self.execute("snapshot"))
        b_snap.grid(row=8, column=0)

        b_link_goban = tk.Checkbutton(self.buttons, text="Save Goban", variable=self.save_goban)
        b_link_goban.grid(row=8, column=1)

        # b_debug = Button(self.buttons, text="Debug", command=lambda: self.checkvar_onoff.set(True))
        # b_debug.grid(row=5, column=0, columnspan=2)

    def build_menu_video(self):
        menu = tk.Menu(self.menubar)
        menu.add_command(label="File (video / image)", command=lambda: self.execute("vidfile"))
        menu.add_command(label="Live video", command=lambda: self.execute("vidlive"))
        self.menubar.insert_cascade(0, label="Input", menu=menu)

    def build_menu_detection(self):
        self.checkvar_detect = tk.BooleanVar()
        self.checkvar_detect.set(True)
        self.radvar_bf = tk.StringVar()
        self.radvar_sf = tk.StringVar()

        menu = tk.Menu(self.menubar)
        menu.add_checkbutton(label="Active", command=self.toggle_active, variable=self.checkvar_detect)
        self.m_board = tk.Menu(menu)
        self.m_stones = tk.Menu(menu)
        menu.add_cascade(label="Board", menu=self.m_board)
        menu.add_cascade(label="Stones", menu=self.m_stones)
        self.menubar.insert_cascade(0, label="Detection", menu=menu)

    def build_menu_training(self):
        # TODO one action to set the samples dest folder
        # TODO one action to open the sampling verification widget
        menu = tk.Menu(self.menubar)
        menu.add_command(label="Random Fill", command=lambda: self.execute("random"))
        menu.add_command(label="Data Generation", command=self.data_gen)
        self.menubar.insert_cascade(0, label="Training", menu=menu)

    def toggle_active(self):
        self.execute("on" if self.checkvar_detect.get() else "off")

    def add_bf(self, bf_class, callback):
        """
        Add the board finder to the menu.

        """
        self.m_board.add_radiobutton(label=bf_class, command=lambda: callback(bf_class),
                                     variable=self.radvar_bf, value=bf_class)

    def add_sf(self, sf_class, callback):
        """
        Add the stones finder to the menu.

        """
        self.m_stones.add_radiobutton(label=sf_class, command=lambda: callback(sf_class),
                                      variable=self.radvar_sf, value=sf_class)

    def select_bf(self, label):
        self.radvar_bf.set(label)
        if label is not None:
            self.checkvar_detect.set(True)

    def select_sf(self, label):
        self.radvar_sf.set(label)
        if label is not None:
            self.checkvar_detect.set(True)

    def video_progress(self, progress):
        self.video_pos.set(progress)

    def data_gen(self):
        window = tk.Toplevel(self)
        nn_frame = DataGeneration(window, snapshot_dir)
        nn_frame.pack()
