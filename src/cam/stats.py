import math

__author__ = 'Kohistan'


def tohisto(mult_factor, values):

    """
    Take an iterable of float values, multiplies them by the factor,
    floor them and store occurrence count in a dict.
    """
    histo = {}
    for val in values:
        intv = int(math.floor(val * mult_factor))
        try:
            histo[intv] += 1
        except KeyError:
            histo[intv] = 1
    return histo