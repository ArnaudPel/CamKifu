import tkinter
from os import listdir, remove
from os.path import join, isfile
from tkinter.ttk import Frame, Label, Button

import numpy as np

from camkifu.config import cvconf
from camkifu.stone.nn_manager import NNManager, TRAIN_DAT_SUFFIX, PNG_SUFFIX
from camkifu.stone.nn_runner import merge_npz, display_histo
from glmain import configure, place, bring_to_front

VERIFY = "Verify"
DELETE = "Delete"

NEW = "New"
SAMPLED = "Sampled"


class Row:
    def __init__(self, master, idx, img_name, button):
        self.manager = master.manager
        self.status = tkinter.StringVar(master=master)
        self.path = join(master.dir, img_name)
        self.idx = idx
        self.button = button
        if isfile(self.path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)):
            self.mark_as_done()
        else:
            self.mark_as_new()

    def gen_data(self):
        x, y = self.manager.gen_data(self.path)
        if x is not None:
            np.savez(self.path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX), X=x, Y=y)
            return True
        return False

    def sample(self):
        if self.gen_data():
            self.mark_as_done()

    def mark_as_done(self):
        self.status.set(SAMPLED)
        self.button["text"] = DELETE
        self.button["command"] = self.delete

    def delete(self):
        try:
            remove(self.path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX))
        except Exception as e:
            print(e)
        self.mark_as_new()

    def mark_as_new(self):
        self.status.set(NEW)
        self.button["text"] = VERIFY
        self.button["command"] = self.sample


class ImgList(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.dir = master.dir
        self.manager = master.manager
        self.rows = []

        for i, img in enumerate(f for f in listdir(master.dir) if f.endswith(PNG_SUFFIX)):
            button = Button(self)
            button.grid(row=i, column=2)
            row = Row(self, i, img, button)
            self.rows.append(row)
            Label(self, text=img).grid(row=i)
            Label(self, textvariable=row.status).grid(row=i, column=1)


class DataGeneration(Frame):

    def __init__(self, master, train_dir):
        Frame.__init__(self, master)
        self.dir = train_dir
        self.manager = NNManager()
        self.img_list = ImgList(self)
        self.img_list.pack()
        Button(self, text='Histo', command=self.histo).pack()
        Button(self, text='Train', command=self.train).pack()

    def train(self):
        # TODO implement and run training in a separate thread
        pass

    def histo(self):
        _, y, _ = merge_npz(self.dir, '(.*\-train data.*)|(arch\d*)')
        display_histo(y)


if __name__ == '__main__':
    root = tkinter.Tk()
    configure(root)

    nn_dir = cvconf.snapshot_dir
    app = DataGeneration(root, nn_dir)
    app.pack()

    place(root)
    bring_to_front()
    root.mainloop()
