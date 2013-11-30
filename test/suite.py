from unittest.result import TestResult
from board import extrapolation

from go import kifu, rules

__author__ = 'Kohistan'

import unittest
import doctest


class MyTestCase(unittest.TestSuite):

    def load_tests(self):
        self.addTest(doctest.DocTestSuite(kifu))
        self.addTest(doctest.DocTestSuite(rules))
        self.addTest(doctest.DocTestSuite(extrapolation))

if __name__ == '__main__':
    suite = MyTestCase()
    suite.load_tests()
    result = TestResult()
    suite.run(result)
    if result.wasSuccessful():
        print "Test suite completed successfully (run=" + str(result.testsRun) + ")."
    else:
        for err in result.errors:
            for line in err:
                print line
        for fail in result.failures:
            for line in fail:
                print line
    #print(result)