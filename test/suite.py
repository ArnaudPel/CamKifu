from unittest.result import TestResult
import go

__author__ = 'Kohistan'

import unittest
import doctest


class MyTestCase(unittest.TestSuite):

    def load_tests(self):
        self.addTest(doctest.DocTestSuite(go.kifu))

if __name__ == '__main__':
    suite = MyTestCase()
    suite.load_tests()
    result = TestResult()
    suite.run(result)
    print(result)