import math
import cv2
import numpy as np
from cam.draw import show

__author__ = 'Kohistan'


def binarize(gray, blocksize=0):
    """
    Probably deprecated because not used.

    """
    bw = cv2.GaussianBlur(gray, (3, 3), 0)
    minsize = min(bw.shape[0], bw.shape[1])
    if blocksize == 0:
        blocksize = max(int(math.floor(minsize / 19)), min(minsize, 19))  # rather arbitrary block ATM

    if blocksize % 2 == 0:
        blocksize += 1

    bw = cv2.adaptiveThreshold(bw, 250, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, 5)

    # additional tuning
    bw = cv2.bitwise_not(bw)
    kernel = np.ones((3, 3), np.uint8)
    # bw = cv2.dilate(bw, kernel, iterations=1)
    # cv2.bitwise_not(bw)

    return bw


def find_black(img):
    """
    Idea : find black zones to remove them from the image.
    """
    nb_splits = 20
    b_width = img.width/nb_splits
    b_height = img.height/nb_splits
    gray = np.zeros((img.width, img.height), np.uint8, 1)
    cv2.cvtColor(img, gray, cv2.COLOR_RGB2GRAY)
    for i in range(nb_splits):
        for j in range(nb_splits):
            if i < 3 or 16 < i or j < 3 or 16 < j:
                subrect = gray[i * b_width:j * b_height, b_width:b_height]
                # todo compute mean pix val and delete black zones
                red = cv2.cv.CV_RGB(255, 0, 0)
                cv2.line(img, (i * b_width, j*b_height), (i * b_width, (j+1)*b_height), red, 2)
                cv2.line(img, (i * b_width, (j+1)*b_height), ((i+1) * b_width, (j+1)*b_height), red, 2)
                cv2.line(img, ((i+1) * b_width, (j+1)*b_height), ((i+1) * b_width, j*b_height), red, 2)
                cv2.line(img, ((i+1) * b_width, j*b_height), (i * b_width, j*b_height), red, 2)
                show(img)