from math import sqrt, floor
import cv2

__author__ = 'Kohistan'


def draw_circles(img, centers, color=(0, 0, 255), radius=5):
    for point in centers:
        point = (int(floor(point[0])), int(floor(point[1])))
        cv2.circle(img, point, radius, cv2.cv.CV_RGB(*color), 1)


def draw_lines(img, segments, color=(0, 255, 0)):
    thickness = 1 + 2*_factor(img)
    for seg in segments:
        if isinstance(seg, Segment):
            p1 = (seg.coords[0], seg.coords[1])
            p2 = (seg.coords[2], seg.coords[3])
        else:
            p1 = (seg[0], seg[1])
            p2 = (seg[2], seg[3])
        colo = cv2.cv.CV_RGB(*color)
        cv2.line(img, p1, p2, colo, thickness=thickness)


def show(img, auto_down=True, name="Camkifu"):
#    screen size, automatic detection seems to be a pain so it is done manually.
    toshow = img
    if auto_down:
        f = _factor(img)
        if f:
            toshow = img.copy()
            name += " (downsized)"
            for i in range(f):
                toshow = cv2.pyrDown(toshow)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, toshow)


def _factor(img):
    """
    Find how many times the image should be "pyrDown" to fit inside the screen.

    """
    width = 1920
    height = 1200
    f = 0
    imwidth = img.shape[1]
    imheight = img.shape[0]
    while width < imwidth or height < imheight:
        f += 1
        imwidth /= 2
        imheight /= 2
    return f


class Segment:

    def __init__(self, seg, img):

        xmid = img.shape[1] / 2
        ymid = img.shape[0] / 2
        xdiff = seg[2] - seg[0]
        ydiff = seg[3] - seg[1]

        self.coords = seg
        if abs(ydiff) < abs(xdiff):
            # horizontal segments (well, segments that are more horizontal than vertical)
            self.slope = float(ydiff) / xdiff
            self.intercept = self.slope * (xmid - seg[0]) + seg[1]
            self.horiz = True
        else:
            # vertical segments
            self.slope = float(xdiff) / ydiff
            self.intercept = self.slope * (ymid - seg[1]) + seg[0]
            self.horiz = False

    def __lt__(self, other):
        return self.intercept < other.intercept

    def __gt__(self, other):
        return self.intercept > other.intercept

    def __str__(self):
        return "Seg(intercept=" + str(self.intercept) + ")"

    def __len__(self):
        x2 = (self.coords[0] - self.coords[2]) ** 2
        y2 = (self.coords[1] - self.coords[3]) ** 2
        return int(sqrt(x2 + y2))








