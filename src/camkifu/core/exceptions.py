

class ControllerWarning(Warning):
    def __init__(self, message=None):
        super().__init__(message)


class CorrectionWarning(Warning):
    """ Related to corrections that the user has made on the Goban (eg. deleted a stone).

    May be used to indicate that the handling of user corrections has not completed normally, eg. algorithms that
    should have learnt from it but didn't for lack of an implementation.

    Attributes:
        corrections: iterable
            The concerned corrections. See StonesFinder.corrected()

    """

    def __init__(self, corrections, message=None):
        if message is None:
            message = "{}"
        correc_str = " ["
        for err, exp in corrections:
            correc_str += "(err:{}, exp:{}), ".format(err, exp)
        correc_str = correc_str[0:-2] + "]"
        super().__init__(str(message) + correc_str)
        self.corrections = corrections


class DeletedError(ValueError):
    """ A stone has been suggested at a location locked because the user has deleted it.

    StonesFinders are invited to adjust their settings accordingly.

    Arguments:
        locations: iterable
            The concerned goban intersections, in numpy coordinates (r, c).
    """

    def __init__(self, locations: 'iterable', message=None):
        if message is None:
            message = "Location has been deleted by user, so it is locked until pixels change significantly."
        super().__init__(message)
        self.locations = locations
