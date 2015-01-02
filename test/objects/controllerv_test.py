from test.objects.controllerv_dev import ControllerVDev
from test.objects.kifuref import KifuChecker

__author__ = 'Arnaud Peloquin'


class ControllerVTest(ControllerVDev):

    def __init__(self, ref_sgf, video=0, vid_bounds=(0, 1), mv_bounds=(0, 1000), failfast=False):
        """
        move_bounds -- The first and last moves of the reference kifu to be checked during testing.
                       Can be useful when processing a few moves inside a bigger game.

        """
        super().__init__(video=video, bounds=vid_bounds)
        # overwrite attribute "kifu" with our checker object
        self.kifu = KifuChecker(ref_sgf, failfast=failfast, bounds=mv_bounds)
        self.init_kifu(mv_bounds)
        self.ignored_instruct = set()

    def init_kifu(self, move_bounds):
        if move_bounds:
            log_ref = self.log  # silence log, in order not to have all the skipped moves printed
            self.log = lambda _: None
            # transfer skipped reference moves to the working structure for consistency
            while self.current_mn < move_bounds[0] - 1:
                self.cvappend(self.kifu.ref.getmove_at(self.current_mn + 1))
            self.log = log_ref

    def pipe(self, instruction, *args):
        try:
            super().pipe(instruction, *args)
        except KeyError:
            if instruction not in self.ignored_instruct:
                print("Unsupported instruction: \"{}\", ignoring.".format(instruction))
                self.ignored_instruct.add(instruction)
