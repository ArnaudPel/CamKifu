import cv2
import numpy as np
from cam.extrapolation import prune, median
from cam.imgutil import draw_circles, draw_lines, show

from cam.hough import find_segments, hough, Grid, runmerge
from config import calibconf

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


def user_corners():
    name = "User Input"
    cam = cv2.VideoCapture(0)
    corners = GridListener()

    calibdata = np.load(calibconf.npfile)
    camera = calibdata[calibconf.camera]
    disto = calibdata[calibconf.distortion]

    while True:
        #try:
            ret, frame = cam.read()
            #ret = True
            #frame = cv2.imread("/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/internet/goban0.jpg")
            if ret:
                cv2.setMouseCallback(name, corners.onmouse)
                frame = cv2.undistort(frame, camera, disto)
                frame = cv2.flip(frame, 1)
                corners.paint(frame)
                show(frame, name=name)
                key = cv2.waitKey(200)
                if key == 113: return
                elif key == 122: corners.undo()
            else:
                print "Camera could not be read."
        #except Exception as ex:
        #    print "Camkifu dropped frame: " + str(ex)


class GridListener():

    def __init__(self, nb=4):
        self.points = []
        self.nb = nb
        self.hull = None

    def onmouse(self, event, x, y, flag, param):
        if not self.ready() and event == cv2.cv.CV_EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        if self.ready():
            self.hull = cv2.convexHull(np.vstack(self.points))

    def undo(self):
        if len(self.points):
            self.points.pop(-1)

    def paint(self, img):
        #draw the clicks
        draw_circles(img, self.points)

        #draw convex hull
        lines = []
        if self.ready():
            for i in range(-1, len(self.hull)-1):
                [[x1, y1]] = self.hull[i]
                [[x2, y2]] = self.hull[i+1]
                lines.append([x1, y1, x2, y2])
            draw_lines(img, lines)

            # draw extrapolated grid
            segs = []
            for i in [-1, 0]:
                p11 = self.hull[i][0]
                p12 = self.hull[i+1][0]
                p21 = self.hull[i+2][0]
                p22 = self.hull[i+3][0]

                size = 18
                for j in range(1, size):
                    x1 = (j*p11[0] + (size-j)*p12[0]) / size
                    x2 = (j*p22[0] + (size-j)*p21[0]) / size
                    y1 = (j*p11[1] + (size-j)*p12[1]) / size
                    y2 = (j*p22[1] + (size-j)*p21[1]) / size
                    segs.append([x1, y1, x2, y2])
            draw_lines(img, segs, color=(50, 120, 50))

    def ready(self):
        return len(self.points) == self.nb

    def __str__(self):
        return "Corners:" + str(self.points)

if __name__ == '__main__':
    #video()
    #picture
    user_corners()























