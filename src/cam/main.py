import cv2
from cam.cframe import user_corners
from cam.extrapolation import median
from cam.imgutil import draw_circles, draw_lines, show

from cam.hough import find_segments, runmerge

__author__ = 'Kohistan'


def picture():
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


def video():
    segment = True
    merge = True
    extrapolate = True
    cam = cv2.VideoCapture(0)
    while True:
        try:
            ret, frame = cam.read()
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if segment:
                grid = find_segments(img)
                for i in range(0):
                    ret, frame = cam.read()
                    if segment: img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    grid += find_segments(img)
                print "segments: " + str(len(grid))

                if merge: grid = runmerge(grid)
                print "segments: " + str(len(grid))

                if extrapolate:
                    grid, discarded = median(grid)
                    draw_lines(frame, discarded.enumerate(), color=(255, 0, 0))

                draw_lines(frame, grid.enumerate())

            show(frame)
            print "looped."
            print
        except Exception as ex:
            print "Camkifu dropped frame: " + str(ex)

        key = cv2.waitKey(20)
        if key == 113:
            return
        elif key == 101:
            extrapolate = not extrapolate
        elif key == 109:
            merge = not merge
            print "merging: " + ("yes" if merge else "no")
        elif key == 115:
            segment = not segment


if __name__ == '__main__':
    user_corners()
























