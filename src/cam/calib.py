"""
    Camera calibration script.
"""
from time import sleep, time

import cv2
import numpy as np

from cam.imgutil import draw_circles, show, VidProcessor
from config import calibconf

__author__ = 'Kohistan'

# todo inline these two guys as parameters, auto-guess, or move to config
nbx = 5  # number of rows
nby = 8  # number of circles per row


def genpattern(x_resol, y_resol):
    """
    Generate a symmetric calibration pattern of circles.

    """
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


class Rectifier(VidProcessor):

    def __init__(self, camera):
        super(self.__class__, self).__init__(camera, None)
        self.camera_coeffs = None
        self.disto = None
        try:
            calibdata = np.load(calibconf.npfile)
            self.camera_coeffs = calibdata[calibconf.camera]
            self.disto = calibdata[calibconf.distortion]
        except IOError or TypeError:
            choice = ""
            while choice not in ("y", "n"):
                choice = raw_input("Would you like to start calibration process ? (y/n)")
            if choice == "y":
                self.mark = time() + calibconf.pause
                self.imagepoints = []
                self.shape = None
                self._calibrate()

    def _calibrate(self):
        # detect a set of calibration patterns in webcam input
        self.run()

        if len(self.imagepoints) == calibconf.shots:
            # fill object points vector
            objectpoints = []
            _, objectref = genpattern(*self.shape)
            for i in range(len(self.imagepoints)):
                objectpoints.append(objectref)
            objectpoints = np.array(objectpoints, dtype=np.float32)
            imagepoints = np.array(self.imagepoints, dtype=np.float32)

            # compute calibration
            retval, cam_mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objectpoints, imagepoints, self.shape)
            if retval:
                print
                print "Calibration completed successfully. Thank you."
                arrays = {calibconf.camera: cam_mtx, calibconf.distortion: dist_coeffs}
                np.savez(calibconf.npfile, **arrays)
                print "Calibration data saved to: " + calibconf.npfile
            else:
                print "Camera calibration failed."
        else:
            print "Calibration has been cancelled."

    def _doframe(self, frame):
        show(frame)
        if self.mark < time():
            self.mark += calibconf.pause
            message = "Could not detect calibration pattern. Please try to place it differently."
            success, shot = cv2.findCirclesGridDefault(frame, (nby, nbx), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
            if success:
                self.imagepoints.append(shot)
                message = "Pattern {0} has been recorded. Please move calibration" \
                          " pattern for next record.".format(len(self.imagepoints))
                self.shape = frame.shape[0:2]
            print message
        if len(self.imagepoints) == calibconf.shots:
            self._done()

    def undistort(self, frame):
        if self.camera_coeffs is not None and self.disto is not None:
            return cv2.undistort(frame, self.camera_coeffs, self.disto)
        return frame


def campattern():
    #noinspection PyArgumentList
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