__author__ = 'Kohistan'


class StateError(Exception):

    def __init__(self, message=None):
        super(StateError, self).__init__(message)