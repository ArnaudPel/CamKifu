import math
import numpy as np
import cv2

__author__ = 'Kohistan'


def draw_circles(img, centers, color=(0, 0, 255)):
    for point in centers:
        point = (int(math.floor(point[0])), int(math.floor(point[1])))
        cv2.circle(img, point, 5, cv2.cv.CV_RGB(*color), 1)


def draw_lines(img, segments, color=(255, 0, 0), thickness=1):
    for seg in segments:
        if isinstance(seg, Segment):
            p1 = (seg.coords[0], seg.coords[1])
            p2 = (seg.coords[2], seg.coords[3])
        else:
            p1 = (seg[0], seg[1])
            p2 = (seg[2], seg[3])
        colo = cv2.cv.CV_RGB(*color)
        cv2.line(img, p1, p2, colo, thickness=thickness)
    #print "found " + str(len(lines)) + " lines"


def show(img, auto_down=True, name="Camkifu"):
#    screen size, automatic detection seems to be a pain so it is done manually.
    width = 1920
    height = 1200
    resized = False
    if auto_down:
        toshow = img.copy()
        while width < toshow.shape[1] or height < toshow.shape[0]:
            toshow = cv2.pyrDown(toshow)
            resized = True
    else:
        toshow = img
    if resized:
        name += " (downsized)"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, toshow)


class Segment:

    def __init__(self, coords, slope, intercept, x_intersect=0, y_intersect=0):
        self.coords = coords
        self.slope = slope
        self.intercept = intercept
        # intersection of the line with an arbitrary horizontal line
        self.x_intersect = x_intersect
        # intersection of the line with an arbitrary vertical line
        self.y_intersect = y_intersect

    def __lt__(self, other):
        return self.intercept < other.intercept

    def __gt__(self, other):
        return self.intercept > other.intercept

    def __str__(self):
        return "Seg(intercept=" + str(self.intercept) + ")"

    #def __eq__(self, other):
    #    return self.intercept == other.intercept and self.slope == other.slope