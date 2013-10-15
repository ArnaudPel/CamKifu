"""
    Camera calibration script.
"""
from time import sleep, time

import cv2
import numpy as np

from cam.imgutil import draw_circles, show
from config import calibconf


__author__ = 'Kohistan'


# todo inline these two guys as parameters, auto-guess, or move to config
nbx = 5  # number of rows  todo swap values
nby = 8  # number of circles per row


def calibrate():
    cam = cv2.VideoCapture(0)
    objectpoints = []
    imagepoints = []
    shape = None

    # detect a set of calibration patterns in webcam input
    mark = time() + calibconf.pause
    while len(imagepoints) < calibconf.shots:
        ret, frame = cam.read()
        if ret:
            show(cv2.flip(frame, 1))  # horizontal flip, because I'm using macOS X camera

            if mark < time():
                mark += calibconf.pause
                message = "Could not detect calibration pattern. Please try to place it differently."
                if ret:
                    success, shot = cv2.findCirclesGridDefault(frame, (nby, nbx), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
                    if success:
                        imagepoints.append(shot)
                        message = "Pattern {0} has been recorded. Please move calibration" \
                                  " pattern for next record.".format(len(imagepoints))
                        shape = frame.shape[0:2]
                print message
        else:
            print "camera read failed."
        if cv2.waitKey(50) == 113: return  # don't try to calibrate if 'q' has been pressed

    # fill object points vector
    _, objectref = genpattern(*shape)
    for i in range(len(imagepoints)):
        objectpoints.append(objectref)
    objectpoints = np.array(objectpoints, dtype=np.float32)
    imagepoints = np.array(imagepoints, dtype=np.float32)

    retval, camera_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, shape)
    if retval:
        print
        print "Calibration completed successfully. Thank you."
        arrays = {calibconf.camera: camera_mtx, calibconf.distortion: dist_coeffs}
        np.savez(calibconf.npfile, **arrays)
        print "Calibration data saved to: " + calibconf.npfile
    else:
        print "Camera calibration failed."


def genpattern(x_resol, y_resol):
    x_border = x_resol / 10
    y_border = y_resol / 10
    xchunk = (x_resol - 2*x_border)/nbx
    ychunk = (y_resol - 2*y_border)/nby

    radius = int(min(xchunk, ychunk) * 0.3)
    centers = []

    # distribute circle centers
    for i in range(nbx):
        for j in range(nby):
            x = x_border + (i+0.5) * xchunk
            y = y_border + (j+0.5) * ychunk
            centers.append((y, x, 0))  # z=0 for homogeneous coordinates

    # draw circles on a new image
    img = np.ones((x_resol, y_resol), np.uint8)
    for i in range(x_resol):
        for j in range(y_resol):
            img[i][j] = 255
    draw_circles(img, centers, color=(255, 0, 0), thickness=-1, radius=radius)
    return img, centers


def campattern():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    tries = 0
    while not ret and tries < 5:
        print "Could not read camera 0."
        ret, frame = cam.read()
        tries += 1
        sleep(5)
    if ret:
        return genpattern(*frame.shape[0:2])
    else:
        print "could not get calibration pattern based on cam resolution."


def urdistort_live():
    calibdata = np.load(calibconf.npfile)
    camera = calibdata[calibconf.camera]
    disto = calibdata[calibconf.distortion]
    print "Camera"
    print camera
    print "Distortion"
    print disto

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if camera is not None and disto is not None:
            frame = cv2.undistort(frame, camera, disto)
            frame = cv2.flip(frame, 1)  # flip only after removing distortion !
            show(frame)
            if cv2.waitKey(50) == 113: return
        else:
            print "No calibration data found here: " + calibconf.npfile
            break


if __name__ == '__main__':
    #calibrate()
    urdistort_live()
    #im, _ = campattern()
    #cv2.imwrite("/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/calib/calib.png", im)






































