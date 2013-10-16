import sys

import cv2
import numpy as np

from cam.imgutil import show, draw_circles, draw_lines, saturate, rgb_histo
from config import calibconf


__author__ = 'Kohistan'


#noinspection PyNoneFunctionAssignment
def user_corners():
    name = "User Input"
    cam = cv2.VideoCapture(0)
    corners = GridListener()

    calibdata = np.load(calibconf.npfile)
    camera = calibdata[calibconf.camera]
    disto = calibdata[calibconf.distortion]

    canoned = False
    mtx = None
    while True:
        ret, frame = cam.read()
        if ret:
            cv2.setMouseCallback(name, corners.onmouse)
            frame = cv2.undistort(frame, camera, disto)
            frame = cv2.flip(frame, 1)

            size = 19 * 30
            if corners.ready() and not canoned:
                src = np.array(corners.hull, dtype=np.float32)
                dst = np.array([(0, 0), (size, 0), (size, size), (0, size)], dtype=np.float32)
                # todo optimization: crop the image around the ROI before computing the transform
                mtx = cv2.getPerspectiveTransform(src, dst)
                canoned = True
            if canoned:
                canon_img = cv2.warpPerspective(frame, mtx, (size, size))
                histo = rgb_histo(canon_img)
                enhanced = saturate(canon_img)
                show(enhanced, name="Enhanced")
                show(histo, name="RGB Histogram")

            corners.paint(frame)
            show(frame, name=name)

            key = cv2.waitKey(200)
            if key == 113:
                return
            elif key == 122:
                corners.undo()
                canoned = False

        else:
            print "Camera could not be read."


class GridListener():
    def __init__(self, nb=4):
        self.points = []
        self.nb = nb
        self.hull = None

    #noinspection PyUnusedLocal
    def onmouse(self, event, x, y, flag, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN and not self.ready():
            self.points.append((x, y))
            if self.ready():
                self._order()

    def undo(self):
        if len(self.points):
            self.points.pop(-1)

    def ready(self):
        return len(self.points) == self.nb

    def paint(self, img):
        #draw the clicks
        draw_circles(img, self.points)

        #draw convex hull
        if self.ready():
            nbpts = len(self.hull) - 1
            color = (0, 0, 255)
            for i in range(-1, nbpts):
                x1, y1 = self.hull[i]
                x2, y2 = self.hull[i + 1]
                draw_lines(img, [[x1, y1, x2, y2]], color)
                color = (255 * (nbpts - i - 1) / nbpts, 0, 255 * (i + 1) / nbpts)

                # draw extrapolated
                #if len(self.hull) == 4:
                #    segs = []
                #    for i in [-1, 0]:
                #        p11 = self.hull[i]
                #        p12 = self.hull[i + 1]
                #        p21 = self.hull[i + 2]
                #        p22 = self.hull[i + 3]
                #
                #size = 18
                #for j in range(1, size):
                #    x1 = (j * p11[0] + (size - j) * p12[0]) / size
                #    x2 = (j * p22[0] + (size - j) * p21[0]) / size
                #    y1 = (j * p11[1] + (size - j) * p12[1]) / size
                #    y2 = (j * p22[1] + (size - j) * p21[1]) / size
                #    segs.append([x1, y1, x2, y2])
                #draw_lines(img, segs, color=(42, 142, 42))

    def _order(self):
        self.hull = []
        idx = 0
        mind = sys.maxint
        if self.ready():
            cvhull = cv2.convexHull(np.vstack(self.points))
            for i in range(len(cvhull)):
                p = cvhull[i][0]
                dist = p[0] ** 2 + p[1] ** 2
                if dist < mind:
                    mind = dist
                    idx = i
            for i in range(idx, idx + len(cvhull)):
                p = cvhull[i % len(cvhull)][0]
                self.hull.append((p[0], p[1]))

    def __str__(self):
        return "Corners:" + str(self.points)