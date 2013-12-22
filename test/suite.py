from unittest import TestSuite, TextTestRunner
from doctest import DocTestSuite
from board import extrapolation

from go import rules

__author__ = 'Kohistan'


def getsuite():
    suite = TestSuite()
    suite.addTest(DocTestSuite(rules))  # todo move this to a DocTestsGolib class

    suite.addTest(DocTestSuite(extrapolation))
    return suite

if __name__ == '__main__':
    runner = TextTestRunner(verbosity=2)
    runner.run(getsuite())