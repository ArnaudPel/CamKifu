import numpy as np

import golib.gui
import test.objects


class ControllerVDev(golib.gui.ControllerBase):
    """
    A no-GUI base controller for "test" code.

    """

    def __init__(self, sgffile=None, video=0, bounds=(0, 1)):
        super().__init__(sgffile=sgffile)
        self.video = video
        self.bounds = bounds
        self.log = lambda _: None
        self.api = {
            "append": self.cvappend,
            "delete": self._delete,
            "bulk": self._bulk_update,
        }

    def pipe(self, instruction, *args):
        """
        Execute instruction straight away on the caller thread.

        """
        self.api[instruction](*args)

    def cvappend(self, move):
        """
        Put the move in the current Rule object, then append it to the kifu.

        """
        move.number = self.head + 1
        self.rules.put(move)
        self._append(move)

    def get_stones(self) -> np.ndarray:
        """
        Return a copy of the current goban state, in the numpy coordinates system.

        """
        return np.array(self.rules.stones, dtype=object).T


class ControllerVTest(ControllerVDev):

    ignored_instruct = set()
    printed_ignored = False

    def __init__(self, ref_sgf, video=0, vid_bounds=(0, 1), mv_bounds=(0, 1000), failfast=False):
        """
        move_bounds -- The first and last moves of the reference kifu to be checked during testing.
                       Can be useful when processing a few moves inside a bigger game.

        """
        super().__init__(video=video, bounds=vid_bounds)
        # overwrite attribute "kifu" with our checker object
        self.kifu = test.objects.KifuChecker(ref_sgf, failfast=failfast, bounds=mv_bounds)
        self.init_kifu(mv_bounds)

        self.api["video_progress"] = self.print_progress
        self.api["select_sf"] = self.check_terminated
        self.last_progress = 0

    def init_kifu(self, move_bounds):
        if move_bounds:
            # fill the working structure with the skipped reference moves, for consistency
            while self.head < move_bounds[0] - 1:
                self.cvappend(self.kifu.ref.getmove_at(self.head + 1))

    def pipe(self, instruction, *args):
        try:
            super().pipe(instruction, *args)
        except KeyError:
            if instruction not in ControllerVTest.ignored_instruct:
                ControllerVTest.ignored_instruct.add(instruction)

    def check_terminated(self, stones_finder):
        if stones_finder is None:
            import sys
            self.print_progress(100)
            sys.stdout.write('\n')
            sys.stdout.flush()
            if not ControllerVTest.printed_ignored:
                message = "ControllerVTest: unsupported instructions {} were ignored"
                print(message.format(ControllerVTest.ignored_instruct))
                ControllerVTest.printed_ignored = True

    def print_progress(self, progress):
        if 3 < progress:  # expect finders to chirrup a bit before displaying the prog bar
            prog_bar(progress, prefix=str(self.video) + ': ')
            self.last_progress = progress


def prog_bar(percent, prefix=''):
    import sys
    p = int(percent)
    sys.stdout.write('\r')
    sys.stdout.write("%s[%-20s] %d%%" % (prefix, '=' * (p // 5), p))
    sys.stdout.flush()
