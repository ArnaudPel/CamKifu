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
        img_path = "/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1.png"
        examples, labels = self.sf.gen_data(img_path)
        np.savez("/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/temp.npz", X=examples, Y=labels)
        # ftemp = np.load("/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/temp.npz")
        # examples = ftemp['X']
        # labels = ftemp['Y']
        file = np.load("/Users/Kohistan/Developer/PycharmProjects/CamKifu/test/camkifu/stone/snapshot-1-train data.npz")
        X = file['X']
        Y = file['Y']
        assert np.array_equal(examples, X), "Input (X) matrix mismatch"
        print("X array seems alright")
        assert np.array_equal(labels, Y),   "Label (Y) matrix mismatch"
        print("Y array seems alright")

if __name__ == '__main__':
    test = TestSfNeural()  # there seems to be a bug with unittest.main(), no will to look into that now
    test.test_should_generate_examples_and_labels()
