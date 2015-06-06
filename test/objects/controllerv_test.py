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

    def __init__(self, ref_sgf, video=0, vid_bounds=(0, 1), mv_bounds=(0, 1000), failfast=False):
        """
        move_bounds -- The first and last moves of the reference kifu to be checked during testing.
                       Can be useful when processing a few moves inside a bigger game.

        """
        super().__init__(video=video, bounds=vid_bounds)
        # overwrite attribute "kifu" with our checker object
        self.kifu = test.objects.KifuChecker(ref_sgf, failfast=failfast, bounds=mv_bounds)
        self.init_kifu(mv_bounds)
        self.ignored_instruct = set()

        self.api["video_progress"] = self.print_progress
        self.last_progress = 0

    def init_kifu(self, move_bounds):
        if move_bounds:
            log_ref = self.log  # silence log, in order not to have all the skipped moves printed
            self.log = lambda _: None
            # transfer skipped reference moves to the working structure for consistency
            while self.head < move_bounds[0] - 1:
                self.cvappend(self.kifu.ref.getmove_at(self.head + 1))
            self.log = log_ref

    def pipe(self, instruction, *args):
        try:
            super().pipe(instruction, *args)
        except KeyError:
            if instruction not in self.ignored_instruct:
                print("Unsupported instruction: \"{}\", ignoring.".format(instruction))
                self.ignored_instruct.add(instruction)

    def print_progress(self, progress):
        if 20 <= progress - self.last_progress:
            print("{0}: {1:.0f}%".format(self.video, progress))
            self.last_progress = progress