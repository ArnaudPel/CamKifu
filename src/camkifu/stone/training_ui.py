import tkinter
from os import listdir, remove
from tkinter.ttk import Frame, Label, Button

from os.path import join, isfile

import numpy as np

from camkifu.stone.tmanager import TRAIN_DAT_SUFFIX, TManager
from glmain import configure, place, bring_to_front


PNG_SUFFIX = ".png"

VERIFY = "Verify"
DELETE = "Delete"

NEW = "New"
SAMPLED = "Sampled"


class Row:
    def __init__(self, nnframe, idx, path, button):
        self.nnframe = nnframe
        self.idx = idx
        self.path = path
        self.status = tkinter.StringVar(master=nnframe)
        self.button = button
        if isfile(self.path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)):
            self.mark_as_done()
        else:
            self.mark_as_new()

    def sample(self):
        self.nnframe.gen_data(self.path)
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


class NnFrame(Frame):
    def __init__(self, master, train_dir):
        Frame.__init__(self, master)
        self.dir = train_dir
        self.sf = TManager()
        self.rows = []

        for i, img in enumerate(f for f in listdir(self.dir) if f.endswith(PNG_SUFFIX)):
            button = Button(self)
            button.grid(row=i, column=2)
            row = Row(self, i, join(self.dir, img), button)
            self.rows.append(row)
            Label(self, text=img).grid(row=i)
            Label(self, textvariable=row.status).grid(row=i, column=1)

    def gen_data(self, img_name):
        fpath = join(self.dir, img_name)
        mpath = fpath.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)
        x, y = self.sf.gen_data(fpath)
        np.savez(mpath, X=x, Y=y)


if __name__ == '__main__':
    root = tkinter.Tk()
    configure(root)

    nn_dir = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training"
    app = NnFrame(root, nn_dir)
    app.pack()

    place(root)
    bring_to_front()
    root.mainloop()
