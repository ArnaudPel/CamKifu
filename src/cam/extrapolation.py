from _bisect import insort, bisect
from cam.hough import Grid

__author__ = 'Kohistan'


def prune(grid, keep=25):
    hsegs = []
    vsegs = []
    for horiz in grid.hsegs:
        insort(hsegs, (len(horiz), horiz))
    for vertic in grid.vsegs:
        insort(vsegs, (len(vertic), vertic))

    longh = hsegs[max(0, len(hsegs) - keep):]
    longv = vsegs[max(0, len(vsegs) - keep):]

    shorth = shortv = []
    if keep < len(hsegs):
        shorth = hsegs[0:len(hsegs) - keep]
    if keep < len(vsegs):
        shortv = vsegs[0:len(vsegs) - keep]

    return Grid([tpl[1] for tpl in longh], [tpl[1] for tpl in longv], grid.img), \
           Grid([tpl[1] for tpl in shorth], [tpl[1] for tpl in shortv], grid.img)


def median(grid):

    # VERTICAL
    #i = len(grid.vsegs) / 2
    #j = i+1
    # median gap
    #gaps = []
    #i = 0
    #for i in range(len(grid.vsegs)-1):
    #    seg0 = grid.vsegs[i]
    #    seg1 = grid.vsegs[i+1]
    #    gap = seg1.intercept - seg0.intercept
    #    assert 0 < gap  # todo rem dev assert
    #    insort(gaps, gap)
    #median_gap = gaps[len(gaps) / 2]
    #
    #lines = []
    #for i in range(len(grid.vsegs)):
    #    seg = grid.vsegs[i]
    #    count = 0
    #    for other_pos in bidirection(grid.vsegs, i, exclude=True):
    #        modulo = abs(seg.intercept - other_pos.intercept) % median_gap
    #        if modulo < median_gap/10 + 1 or 9*median_gap/10 - 1 < modulo:
    #            count += 1
    #            if count == 4:
    #                break
    #    if count == 4:
    #        lines.append(seg)

    # median slope
    slopes = []
    for seg in grid.hsegs:
        insort(slopes, seg.slope)
    median_slope = slopes[len(slopes) / 2]

    # keep lines having a good slope only
    parallels = []
    for hseg in grid.hsegs:
        if median_slope - 0.015 < hseg.slope < median_slope + 0.015:
            parallels.append(hseg)

    # median gap for remaining lines
    gaps = []
    for i in range(len(parallels) - 1):
        gap = parallels[i + 1].intercept - parallels[i].intercept
        insort(gaps, gap)
        assert 0 < gap  # todo remove dev assert
    median_gap = float(gaps[len(gaps) / 2])  # float for later divisions
    if median_gap < 12:
        print "warning, median gap seems very small:" + str(median_gap)

    occurrence = 4
    positions = [i for i in range(len(parallels))]
    good_pos = set()
    discarded = []
    while 0 < len(positions):
        i = len(positions)/2
        pos = positions.pop(i)
        seg = parallels[pos]
        brothers = []
        for other_pos in bidirection(parallels, pos, exclude=True):
            oseg = parallels[other_pos]
            gap = abs(seg.intercept - oseg.intercept)  # todo can we remove abs() ?
            modulo = gap % median_gap
            if modulo < median_gap / 10 or (9 * median_gap) / 10 < modulo:
                brothers.append(other_pos)
                if len(brothers) == occurrence:
                    break
        if len(brothers) == occurrence:
            good_pos.add(pos)
            for bro in brothers:
                good_pos.add(bro)
                try:
                    positions.remove(bro)  # this guy has good friends, no need to test it
                except ValueError:
                    pass  # expected
        else:
            discarded.append(parallels[pos])

    return Grid([parallels[x] for x in good_pos], grid.vsegs, grid.img),\
            Grid(discarded, [], grid.img)
    #return Grid(parallels, grid.vsegs, grid.img)


def bidirection(alist, start, exclude=False):
    """
    >>> alist = [1,2,3,4,5,6,7,8,9]
    >>> print [x for x in bidirection(alist, 3)]
    [4, 3, 5, 2, 6, 1, 7, 8, 9]
    >>> print [x for x in bidirection(alist, 0)]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> print [x for x in bidirection(alist, len(alist) - 1)]
    [9, 8, 7, 6, 5, 4, 3, 2, 1]
    >>> print [x for x in bidirection(alist, 3, exclude=True)]
    [5, 3, 6, 2, 7, 1, 8, 9]
    """
    bw = start - 1
    fw = start
    if exclude:
        fw += 1
    forward = fw < len(alist)
    while 0 <= bw or fw < len(alist):
        if forward:
            yield fw
            fw += 1
            forward = bw < 0
        else:
            yield bw
            bw -= 1
            forward = fw < len(alist)














