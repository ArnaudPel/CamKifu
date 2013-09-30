from _bisect import insort
from cam.hough import Grid

__author__ = 'Kohistan'


def prune(grid):
    hsegs = []
    vsegs = []
    for horiz in grid.hsegs:
        insort(hsegs, (len(horiz), horiz))
    for vertic in grid.vsegs:
        insort(vsegs, (len(vertic), vertic))

    longh = hsegs[max(0, len(hsegs)-19):]
    longv = vsegs[max(0, len(vsegs)-19):]

    shorth = shortv = []
    if 19 < len(hsegs):
        shorth = hsegs[0:len(hsegs)-19]
    if 19 < len(vsegs):
        shortv = vsegs[0:len(vsegs)-19]

    return Grid([tpl[1] for tpl in longh], [tpl[1] for tpl in longv], grid.img),\
           Grid([tpl[1] for tpl in shorth], [tpl[1] for tpl in shortv], grid.img)