from Tkconstants import LEFT
from Tkinter import Frame, Label, Tk
from go.kifu import Kifu
from difflib import SequenceMatcher

__author__ = 'Arnaud Peloquin'


class KifuChecker(Kifu):
    """
    A Kifu holding another (reference) Kifu.
    Aim to provides fail-fast or diff tools to test and analyse detection algorithms findings.

    """

    def __init__(self, reffile, sgffile=None, failfast=False, bounds=(0, 1000)):
        Kifu.__init__(self, sgffile=sgffile)
        self.ref = Kifu(sgffile=reffile)
        self.failfast = failfast
        self.bounds = bounds
        if failfast:
            self.check()

    def check(self):
        if self.failfast:
            idx = 1
            ref_mv = self.ref.getmove_at(idx)
            while ref_mv:
                # fail at first move difference
                mv = self.getmove_at(idx)
                if mv != ref_mv:
                    msg = "At move {0}: expected {1}, got {2}.  (failfast mode is on)".format(idx, ref_mv, mv)
                    raise AssertionError(msg)
                idx += 1
                ref_mv = self.ref.getmove_at(idx)

        # extract diff data between move sequences
        f, l = self.bounds
        ref = self.ref.get_move_seq(first=f, last=l)
        seq = self.get_move_seq(first=f, last=l)
        return SequenceMatcher(a=ref, b=seq)


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
        miss += concat(matcher.a, idx1, block[0])
        idx1 = block[0] + block[2]

        # display items that are not present in ref (or at least not at this pos)
        unex += concat(matcher.b, idx2, block[1])
        idx2 = block[1] + block[2]

        # keep vertical alignment
        mlen = max(len(unex), len(miss))
        good = good.ljust(mlen)
        unex = unex.ljust(mlen)
        miss = miss.ljust(mlen)

        # display matching items
        good += concat(matcher.a, block[0], block[0] + block[2])
        unex = unex.ljust(len(good))
        miss = miss.ljust(len(good))
    print "Matched   : " + good
    print "Missed    : " + miss
    print "Unexpected: " + unex
    print "Ratio     : " + str(matcher.ratio())


def display_matcher(matcher, master=None):
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
        miss = concat(matcher.a, idx1, block[0])
        Label(master=sequence, text=miss, fg="red").pack(side=LEFT)
        idx1 = block[0] + block[2]

        # display items that are not present in ref (or at least not at this pos)
        unex = concat(matcher.b, idx2, block[1])
        Label(master=sequence, text=unex, fg="dark gray").pack(side=LEFT)
        idx2 = block[1] + block[2]

        # display matching items
        match = concat(matcher.a, block[0], block[0] + block[2])
        Label(master=sequence, text=match, fg="dark green").pack(side=LEFT)
    return sequence


def concat(seq, start, end):
    """
    Utility to display a portion of a sequence as a string, and replace with a count when the portion is too long.

    start -- index of first element to concatenate, inclusive.
    end -- index of last element to concatenate, exclusive.
    seq -- contains the elements.

    """
    string = ""
    if 4 < end - start:
        string += str(seq[start])
        string += " .. (%s more) .. " % (end - start - 2)
        string += str(seq[end - 1])
    else:
        for i in range(start, end):
            string += str(seq[i]) + " "
    return string