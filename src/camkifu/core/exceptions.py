__author__ = 'Arnaud Peloquin'


class PipeWarning(Warning):

    def __init__(self, message=None):
        super().__init__(message)


class CorrectionWarning(Warning):
    """
    The treatment of corrections that the user has made on the Goban (eg. deleted a stone) by a VidProcessor has not
    completed normally. This instance holds the list of concerned corrections.

    """

    def __init__(self, corrections, message=None):
        if message is None:
            message = "{}"
        correcs = " ["
        for err, exp in corrections:
            correcs += "(err:{}, exp:{}), ".format(err, exp)
        correcs = correcs[0:-2] + "]"
        super().__init__(str(message) + correcs)
        self.corrections = corrections


class DeletedError(ValueError):
    """
    To be raised when a stone has been suggested at a location that:
        - has recently been deleted by user
        - has not changed much (pixel-wise) since

    So the stone suggestion is most likely pollution and is refused.
    StonesFinder are invited to adjust their settings accordingly if possible.

    """

    def __init__(self, locations, message=None):
        """
        locations -- the (r, c) locations that have been deleted and should not be suggested.

        """
        if message is None:
            message = "Location has been deleted by user, so it is locked until pixels change significantly."
        super().__init__(message)
        self.locations = locations
