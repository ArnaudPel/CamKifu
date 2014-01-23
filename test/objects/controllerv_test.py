from gui.controller import ControllerBase
from test.objects.kifuref import KifuChecker

__author__ = 'Arnaud Peloquin'


class ControllerVTest(ControllerBase):

    def __init__(self, reffile, sgffile=None, video=0, vid_bounds=(0, 1), failfast=False, move_bounds=(0, 1000)):
        """
        move_nr -- specify the first move of the reference kifu to be used when testing.
                    can be useful when processing a few moves inside a bigger game.

        """
        super(ControllerVTest, self).__init__()  # the kifu will be overridden by our checker object
        self.kifu = KifuChecker(reffile, sgffile=sgffile, failfast=failfast, bounds=move_bounds)
        self.video = video
        self.bounds = vid_bounds

        if move_bounds and (sgffile is None):
            # transfer skipped reference moves to the working structure for consistency
            while self.current_mn < move_bounds[0] - 1:
                self.cvappend(self.kifu.ref.getmove_at(self.current_mn + 1))

        self.api = {"append": self.cvappend,
                    "bfinder": self.add_finder,
                    "sfinder": self.add_finder}

    def cvappend(self, move):
        move.number = self.current_mn + 1
        self.rules.put(move)
        self._append(move)

    @staticmethod
    def add_finder(f_class, callback, select=False):
        """
        This method is a "plug" of its GUI-version, hence the non-intuitive syntax.

        """
        if select:
            callback(f_class)

    def pipe(self, instruction, args):
        """
        Run instruction on the current thread (i.e. caller's thread),
        because we assume that we are running without GUI.
        """
        self.api[instruction](*args)