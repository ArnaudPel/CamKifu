import tkinter
from os import listdir
from tkinter.ttk import Frame, Entry, Label

from os.path import join, isfile

from camkifu.stone.sf_neural import TRAIN_DAT_SUFFIX
from glmain import configure, place, bring_to_front

PNG_SUFFIX = ".png"


class NnFrame(Frame):

    def __init__(self, master, train_dir):
        Frame.__init__(self, master)
        self.dir = train_dir

        for i, pic in enumerate(f for f in listdir(self.dir) if f.endswith(PNG_SUFFIX)):
            Label(self, text=pic).grid(row=i)
            mat = pic.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)
            if isfile(join(self.dir, mat)):
                Label(self, text="Sampled").grid(row=i, column=1)
            else:
                Label(self, text="New").grid(row=i, column=1)

if __name__ == '__main__':
    root = tkinter.Tk()
    configure(root)

    nn_dir = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training"
    app = NnFrame(root, nn_dir)
    app.pack()

    place(root)
    bring_to_front()
    root.mainloop()

