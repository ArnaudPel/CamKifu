import math
import cv2
import numpy as np

from cam.hough import find_squares, hough
from cam.prepare import binarize
from cam.draw import _show, draw_lines
from cam.stats import tohisto
from gui.plot import plot_histo

__author__ = 'Kohistan'


def find_grid(img):
    # todo implement
    pass


def extrapolate(lines):

    """
    Detects lines distributed periodically in the space, remove pollutions
    and create undetected lines where they should be. The method also tries to
    determine a better bounding box around the goban, in order to crop the picture.
    """

    pruned = []
    mult_factor = 1000
    slopes = []
    for line in lines:
        slopes.append(line.slope)
    histo = tohisto(mult_factor, slopes)
    plot_histo(histo, "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/histo.png")
    best_occ = 0
    best_slope = None
    for (slope, occurrence) in histo.iteritems():
        if best_occ < occurrence:
            best_occ = occurrence
            best_slope = slope
    if best_slope is not None:
        for line in lines:
            if int(math.floor(line.slope * 1000) == best_slope):  # todo compute perspective variations
                pruned.append(line)
    return pruned


if __name__ == '__main__':

    filename = "empty cut.jpg"

    path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/original/" + filename
    img = cv2.imread(path, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # find_black(img)
    # find_grid(img)

    find_squares(img)

    cv2.waitKey()
    cv2.destroyAllWindows()























