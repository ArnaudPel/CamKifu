import math

import cv2
import numpy as np

from camkifu.stone.nn_manager import NNManager
from golib.config.golib_conf import B

__author__ = 'Arnaud Peloquin'


class TestTManager:

    def __init__(self):
        self.sf = NNManager()

    # https://docs.python.org/3/library/unittest.html#unittest.TestLoader.testMethodPrefix
    def should_compute_stones_from_labels(self):
        assert list(self.sf.compute_stones(27)) == ['E', 'E', 'E', 'B']
        assert list(self.sf.compute_stones(36)) == ['E', 'E', 'B', 'B']
        assert list(self.sf.compute_stones(64)) == ['B', 'E', 'B', 'W']

    def should_compute_class_indices(self):
        assert list(self.sf.class_indices()[3, 2]) == [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                                       70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
        assert list(self.sf.class_indices()[0, 2]) == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50,
                                                       53, 56, 59, 62, 65, 68, 71, 74, 77, 80]

    def labels_should_match_inputs(self):
        file = np.load("/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1-train data.npz")
        Xref = file['X']
        newdim = math.sqrt(Xref.shape[1])
        img = Xref.reshape(Xref.shape[0], newdim, newdim)
        Yref = file['Y']
        for i in range(Xref.shape[0]):
            cv2.imshow('Test SF Neural', img[i])
            print(self.sf.compute_stones(np.argmax(Yref[i])))
            if chr(cv2.waitKey()) == 'q':
                break


if __name__ == '__main__':
    test = TestTManager()  # there seems to be a bug with unittest.main(), no will to look into that now
    # test.test_should_generate_examples_and_labels()
    # test.labels_should_match_inputs()
    test.should_compute_class_indices()
    for lab in [80]:
        print("{} -> {}".format(lab, test.sf.compute_stones(lab)))
    stones = np.ndarray((2, 2), dtype=object)
    stones[:] = B
    print("{} -> {}".format(stones, test.sf.compute_label(0, 2, 0, 2, stones)))
    pass
