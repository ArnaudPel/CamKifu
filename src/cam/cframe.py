import cv2
import numpy as np
import sys
from cam.imgutil import show, draw_circles, draw_lines
from config import calibconf

__author__ = 'Kohistan'


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
        #ret = True
        #frame = cv2.imread("/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/pics/internet/goban0.jpg")

        if ret:
            cv2.setMouseCallback(name, corners.onmouse)
            frame = cv2.undistort(frame, camera, disto)
            frame = cv2.flip(frame, 1)
            corners.paint(frame)
            show(frame, name=name)
            key = cv2.waitKey(200)

            if corners.ready() and not canoned:
                mtx = canonical_rot(corners.hull, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
                print mtx
                canoned = True
            if canoned:
                transformed = []
                for extra in corners.extrapoints:
                    t = mtx.dot(extra)
                    transformed.append(t * 500 / t[2])
                canon_img = np.zeros((500, 500), dtype=np.uint8)
                draw_circles(canon_img, transformed)
                show(canon_img, "Canonical Space")

            if key == 113:
                return
            elif key == 122:
                corners.undo()
                canoned = False

        else:
            print "Camera could not be read."


def canonical_rot(image_points, canonical_points):
    """
    Compute the transformation that brings the goban plane into the canonical frame
    in order to have it in an more Euclidean environment.
    Return A such that: A.P = p
    Where 'P' is a point in the image frame, and 'p' is the corresponding point in the canonical frame.

    """
    assert len(image_points) == 4
    assert len(canonical_points) == 4
    coeffs = []
    values = []
    for i in range(4):
        imp = image_points[i]       # X  Y  Z
        cap = canonical_points[i]   # x  y  z

        #  X  Y  Z  0  0  0  -X.x  -Y.x
        coeffs.append([imp[0], imp[1], imp[2], 0, 0, 0, -imp[0] * cap[0], -imp[1] * cap[0]])
        #  0  0  0  X  Y  Z  -X.y  -Y.y
        coeffs.append([0, 0, 0, imp[0], imp[1], imp[2], -imp[0] * cap[1], -imp[1] * cap[1]])

        # Z.x
        values.append(imp[2] * cap[0])
        # Z.y
        values.append(imp[2] * cap[1])
    # return the matrix that transforms image points to canonical frame points
    sol = np.linalg.solve(coeffs, values)
    return np.array([
          [sol[0], sol[1], sol[2]]
        , [sol[3], sol[4], sol[5]]
        , [sol[6], sol[7], 1]
    ])


class GridListener():
    def __init__(self, nb=4):
        self.points = []
        self.nb = nb
        self.hull = None
        self.extrapoints = []

    def onmouse(self, event, x, y, flag, param):
        if event == cv2.cv.CV_EVENT_LBUTTONDOWN:
            if not self.ready():
                self.points.append((x, y))
                if self.ready():
                    self._order()
            else:
                self.extrapoints.append((x, y, 1))

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
                x1, y1, _ = self.hull[i]
                x2, y2, _ = self.hull[i + 1]
                draw_lines(img, [[x1, y1, x2, y2]], color)
                color = (255 * (nbpts - i-1) / nbpts, 0, 255 * (i+1) / nbpts)

            # draw extrapolated
            if len(self.hull) == 4:
                segs = []
                for i in [-1, 0]:
                    p11 = self.hull[i]
                    p12 = self.hull[i + 1]
                    p21 = self.hull[i + 2]
                    p22 = self.hull[i + 3]

                    size = 18
                    for j in range(1, size):
                        x1 = (j * p11[0] + (size - j) * p12[0]) / size
                        x2 = (j * p22[0] + (size - j) * p21[0]) / size
                        y1 = (j * p11[1] + (size - j) * p12[1]) / size
                        y2 = (j * p22[1] + (size - j) * p21[1]) / size
                        segs.append([x1, y1, x2, y2])
                draw_lines(img, segs, color=(42, 142, 42))

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
                self.hull.append((p[0], p[1], 1))

    def __str__(self):
        return "Corners:" + str(self.points)