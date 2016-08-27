import unittest

import numpy as np

from camkifu.stone.sf_neural import SfNeural

__author__ = 'Arnaud Peloquin'


class TestSfNeural(unittest.TestCase):

    def __init__(self):
        super().__init__()
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

if __name__ == '__main__':
    test = TestSfNeural()  # there seems to be a bug with unittest.main(), no will to look into that now
    test.test_should_generate_examples_and_labels()
    # test.should_compute_stones_from_labels()
