import tkinter
from os import listdir
from tkinter.ttk import Frame, Label, Button

from os.path import join, isfile

import numpy as np

from camkifu.stone.sf_neural import TRAIN_DAT_SUFFIX, SfNeural
from glmain import configure, place, bring_to_front

DELETE = "Delete"

PNG_SUFFIX = ".png"
SAMPLED = "Sampled"
NEW = "New"


class NnFrame(Frame):
    def __init__(self, master, train_dir):
        Frame.__init__(self, master)
        self.dir = train_dir
        self.sf = SfNeural(None)
        self.blist = []

        for i, img in enumerate(f for f in listdir(self.dir) if f.endswith(PNG_SUFFIX)):
            status = tkinter.StringVar(master=self)
            if isfile(join(self.dir, img.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX))):
                status.set(SAMPLED)
                btext = DELETE

                def bcommand(idx=i):
                    print("Todo - butt" + str(idx))
            else:
                status.set(NEW)
                btext = "Verify"

                def bcommand(n=img, s=status, idx=i):
                    self.gen_data(n, s, idx)
            button = Button(self, text=btext, command=bcommand)
            button.grid(row=i, column=2)
            self.blist.append(button)

            Label(self, text=img).grid(row=i)
            Label(self, textvariable=status).grid(row=i, column=1)

    def gen_data(self, img_name, svar, idx):
        fpath = join(self.dir, img_name)
        mpath = fpath.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)
        x, y = self.sf.gen_data(fpath)
        np.savez(mpath, X=x, Y=y)
        svar.set(SAMPLED)
        self.blist[idx]["text"] = DELETE


if __name__ == '__main__':
    root = tkinter.Tk()
    configure(root)

    nn_dir = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training"
    app = NnFrame(root, nn_dir)
    app.pack()

    place(root)
    bring_to_front()
    root.mainloop()
