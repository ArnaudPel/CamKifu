import math

import cv2
import numpy as np

from camkifu.core.imgutil import show
from camkifu.stone.sf_neural import SfNeural

__author__ = 'Arnaud Peloquin'


class TestSfNeural:

    def __init__(self):
        self.sf = SfNeural(None)

    # https://docs.python.org/3/library/unittest.html#unittest.TestLoader.testMethodPrefix
    def test_should_generate_examples_and_labels(self):
        # print("Use previously saved matrix ?")
        # load = input() == 'y'
        # prev_mat_file = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1-train data.npz"
        # if not load:
        img_path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1.png"
        examples, labels = self.sf.gen_data(img_path)
        #     np.savez(prev_mat_file, X=examples, Y=labels)
        # else:
        #     ftemp = np.load(prev_mat_file)
        #     examples = ftemp['X']
        #     labels = ftemp['Y']
        file = np.load("/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1-train data.npz")
        Xref = file['X']
        Yref = file['Y']
        assert np.array_equal(examples, Xref), "Input (X) matrix mismatch"
        print("X array seems alright")
        assert np.array_equal(labels, Yref),   "Label (Y) matrix mismatch"
        print("Y array seems alright")

    def should_compute_stones_from_labels(self):
        assert self.sf.compute_stones(27) == ['E', 'E', 'E', 'B']
        assert self.sf.compute_stones(36) == ['E', 'E', 'B', 'B']
        assert self.sf.compute_stones(64) == ['B', 'E', 'B', 'W']

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

    def labels_histo(self, train_file, nb_bins=3 ** 4):
        labels = np.load(train_file)['Y']
        y = np.zeros((labels.shape[0], 1), dtype=np.uint8)
        for i, label_vect in enumerate(labels):
            y[i] = np.argmax(label_vect)
        hist_item = cv2.calcHist([y], [0], None, [nb_bins], [1, nb_bins])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        zoom = 7
        chart_width = nb_bins * zoom
        bins = np.arange(0, chart_width, zoom).reshape(nb_bins, 1)
        pts = np.column_stack((bins, hist))
        canvas = np.zeros((300, chart_width, 3))
        cv2.polylines(canvas, [pts], False, (255, 255, 255))
        histo = np.flipud(canvas)
        show(histo, name='Training data labels distribution')
        cv2.waitKey()


if __name__ == '__main__':
    test = TestSfNeural()  # there seems to be a bug with unittest.main(), no will to look into that now
    # test.test_should_generate_examples_and_labels()
    # test.labels_should_match_inputs()
    test.labels_histo("/Users/Kohistan/Developer/PycharmProjects/CamKifu/res/temp/training/all train.npz")
