from Tkconstants import LEFT
from Tkinter import Frame, Label, Tk
from go.kifu import Kifu
from difflib import SequenceMatcher

__author__ = 'Kohistan'


class KifuChecker(Kifu):
    """
    A Kifu holding another (reference) Kifu.
    Aim to provides fail-fast or diff tools to test and analyse detection algorithms findings.

    """

    def __init__(self, reffile, sgffile=None, failfast=False):
        Kifu.__init__(self, sgffile=sgffile)
        self.ref = Kifu(sgffile=reffile)
        self.failfast = failfast
        if failfast:
            self.check()

    def check(self):
        # fail at first move difference
        if self.failfast:
            idx = 1
            ref_mv = self.ref.getmove_at(idx)
            while ref_mv:
                mv = self.getmove_at(idx)
                if mv != ref_mv:
                    msg = "At move {0}: expected {1}, got {2}".format(idx, ref_mv, mv)
                    raise AssertionError(msg)
                idx += 1
                ref_mv = self.ref.getmove_at(idx)

        # extract diff data between move sequences
        else:
            return SequenceMatcher(a=(self.ref.get_main_seq()), b=(self.get_main_seq()))


def print_matcher(matcher):
    """
    Print data from matcher to the console.
    Interpretation:
    -- matcher.a is the reference.
    -- matcher.b is the sequence under test.

    """
    blocks = matcher.get_matching_blocks()
    good = unex = miss = ""
    idx1 = idx2 = 0
    for block in blocks:
        # display items from reference that have been missed
        for i in range(idx1, block[0]):
            miss += str(matcher.a[i]) + " "
        idx1 = block[0] + block[2]

        # display items that are not present in ref (or at least not at this pos)
        for i in range(idx2, block[1]):
            unex += str(matcher.b[i]) + " "
        idx2 = block[1] + block[2]

        # keep vertical alignment
        mlen = max(len(unex), len(miss))
        good = good.ljust(mlen)
        unex = unex.ljust(mlen)
        miss = miss.ljust(mlen)

        # display matching items
        for i in range(block[0], block[0]+block[2]):
            good += str(matcher.a[i]) + " "
        unex = unex.ljust(len(good))
        miss = miss.ljust(len(good))
    print "Matched   : " + good
    print "Missed    : " + miss
    print "Unexpected: " + unex
    print "Ratio     : " + str(matcher.ratio())


def tk_sequence(matcher, master=None):
    """
    Return a Tkinter display of the matcher information.
    Interpretation:
    -- matcher.a is the reference.
    -- matcher.b is the sequence under test.

    """
    sequence = Frame(master=master)
    idx1 = idx2 = 0
    blocks = matcher.get_matching_blocks()
    for block in blocks:
        # display items from reference that have been missed
        miss = ""
        for i in range(idx1, block[0]):
            miss += str(matcher.a[i]) + " "
        Label(master=sequence, text=miss, fg="red").pack(side=LEFT)
        idx1 = block[0] + block[2]

        # display items that are not present in ref (or at least not at this pos)
        unex = ""
        for i in range(idx2, block[1]):
            unex += str(matcher.b[i]) + " "
        Label(master=sequence, text=unex, fg="gray").pack(side=LEFT)
        idx2 = block[1] + block[2]

        # display matching items
        match = ""
        for i in range(block[0], block[0] + block[2]):
            match += str(matcher.a[i]) + " "
        Label(master=sequence, text=match, fg="green").pack(side=LEFT)
    return sequence


if __name__ == '__main__':
    ref = "/Users/Kohistan/Documents/Go/Perso Games/reference.sgf"
    sgf = "/Users/Kohistan/Documents/Go/Perso Games/experience.sgf"
    checker = KifuChecker(reffile=ref, sgffile=sgf)
    m = checker.check()
    print_matcher(m)

    root = Tk()
    frame = tk_sequence(m, root)
    frame.pack()
    root.mainloop()






















