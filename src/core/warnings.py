__author__ = 'Kohistan'


class PipeWarning(Warning):

    def __init__(self, message=None):
        super(PipeWarning, self).__init__(message)