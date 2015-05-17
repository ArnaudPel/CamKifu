import ckmain
import test.objects


"""
Script that can be used to run all vision on the main thread. The counterpart is that no Tkinter
GUI can be used, as it has to monopolize the main thread.

This is mainly useful when in need to display something and pause in the middle of a vision algo,
especially to use waitKey().

"""

def main(video=0, sgf=None, bounds=(0, 1), bf=None, sf=None):
    # run in dev mode, everything on the main thread
    controllerv_dev = test.objects.ControllerVDev(sgffile=sgf, video=video, bounds=bounds)
    vision = test.objects.VManagerSeq(controllerv_dev, bf=bf, sf=sf)
    vision.run()

if __name__ == '__main__':
    args = ckmain.get_argparser().parse_args()
    main(video=args.video, sgf=args.sgf, bounds=args.bounds, bf=args.bf, sf=args.sf)
