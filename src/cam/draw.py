import math
import numpy as np
import cv2

__author__ = 'Kohistan'


def draw_circles(img, centers, color=(0, 0, 255)):
    for point in centers:
        point = (int(math.floor(point[0])), int(math.floor(point[1])))
        cv2.circle(img, point, 5, cv2.cv.CV_RGB(*color), 1)


def _show(img, auto_down=True, name="Camkifu"):
#    screen size, automatic detection seems to be a pain so it is done manually.
    width = 1920
    height = 1200
    if auto_down:
        toshow = img.copy()
        while width < toshow.shape[0] or height < toshow.shape[1]:
            toshow = cv2.pyrDown(toshow)
    else:
        toshow = img
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, toshow)


def draw_lines(img, lines, color=(255, 0, 0)):
    for line in lines:
        if isinstance(line, Segment):
            p1 = line.coords[0:1]
            p2 = line.coords[2:3]
        else:
            p1 = (line[0], line[1])
            p2 = (line[2], line[3])
        colo = cv2.cv.CV_RGB(*color)
        cv2.line(img, p1, p2, colo, thickness=1)
    #print "found " + str(len(lines)) + " lines"


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

    #def __eq__(self, other):
    #    return self.intercept == other.intercept and self.slope == other.slope