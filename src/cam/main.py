import math
import cv2
from cam.draw import draw_lines, show

from cam.regression import merge
from cam.hough import find_segments
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

    filename = "4 stones.jpg"

    path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/original/" + filename
    img = cv2.imread(path, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)

    hlines, vlines = find_segments(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    segimg = img.copy()

    draw_lines(segimg, hlines, color=(0, 255, 0), thickness=1)
    merged, discarded = merge(hlines)
    draw_lines(img, merged, thickness=1)

    draw_lines(segimg, vlines, color=(0, 255, 0), thickness=1)
    merged2, discarded2 = merge(vlines)
    draw_lines(img, merged2, thickness=1)

    #draw_lines(img, discarded, thickness=1)
    show(segimg, name="Segments")
    show(img, name="Merged")

    key = cv2.waitKey()
    cv2.destroyAllWindows()























