import cv2
from cam.calib import Rectifier
from cam.stones import StonesFinder
from cam.extrapolation import median
from cam.imgutil import draw_circles, draw_lines, show

from cam.board import find_segments, runmerge, BoardFinder

__author__ = 'Kohistan'


def picture():
    """
    Old code that is most likely deprecated, or simply useless.

    """
    filename = "original/fenetre2.jpg"

    path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/" + filename
    img = cv2.imread(path, flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)

    grid = find_segments(img)
    grid = runmerge(grid)
    grid, discarded = median(grid)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    draw_lines(img, discarded.enumerate(), color=(255, 0, 0))
    draw_lines(img, grid.enumerate())

    xmid = img.shape[1] / 2
    ymid = img.shape[0] / 2
    draw_circles(img, [(xmid, seg.intercept) for seg in grid.hsegs])
    draw_circles(img, [(seg.intercept, ymid) for seg in grid.vsegs])

    show(img, name=filename)
    if cv2.waitKey() == 113: return
    cv2.destroyAllWindows()


def main():
    #noinspection PyArgumentList
    cam = cv2.VideoCapture(0)

    rectifier = Rectifier(cam)
    board_finder = BoardFinder(cam, rectifier)
    stones_finder = StonesFinder(cam, rectifier)

    states = {"plain": 0, "canonical": 1}
    state = 0

    while True:
        if state == states["plain"]:
            board_finder.run()
            stones_finder.transform = board_finder.mtx
            stones_finder.size = board_finder.size
            if board_finder.mtx is not None:
                state = 1
            else:
                break

        elif state == states["canonical"]:
            stones_finder.run()
            if stones_finder.undo:
                board_finder.perform_undo()
                state = 0
                stones_finder.undo = False
            else:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()