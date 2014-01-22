from unittest import TestSuite, TextTestRunner
from doctest import DocTestSuite

from go import rules

__author__ = 'Kohistan'


def getsuite():
    suite = TestSuite()
    suite.addTest(DocTestSuite(rules))  # todo move this to a DocTestsGolib class
    return suite

if __name__ == '__main__':
    runner = TextTestRunner(verbosity=2)
    runner.run(getsuite())