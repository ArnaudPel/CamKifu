import cv2
from cam.draw import draw_lines, show
from cam.extrapolation import prune

from cam.regression import runmerge
from cam.hough import find_segments, hough

__author__ = 'Kohistan'


def find_grid(img):

    grid = find_segments(img)
    grid = runmerge(grid)
    draw_lines(img, grid.enumerate())
    show(img)
    if cv2.waitKey() == 113: return
    cv2.destroyAllWindows()


if __name__ == '__main__':

    filename = "original/salon2.jpg"

    path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/" + filename
    im = cv2.imread(path, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    find_grid(im)
























