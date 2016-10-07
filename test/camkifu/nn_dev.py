import argparse
import math

import cv2
import numpy as np

from camkifu.config import cvconf
from camkifu.core.imgutil import show
from camkifu.stone.nn_manager import NNManager
from camkifu.stone.nn_runner import split_data


def visualize_l0(manager, relu=True):
    filters = manager.get_net().get_weights()[0]
    nb_filt = filters.shape[3]
    nb_rows, nb_cols = 1, nb_filt
    # find the couple of dividers closest to sqrt
    for d in reversed(range(int(math.sqrt(nb_filt)))):
        if nb_filt % d == 0:
            nb_rows, nb_cols = d, nb_filt // d
            break
    depth = 3
    zoom = 8
    margin = 6
    img_height = (filters.shape[0] * zoom + margin) * nb_rows - margin
    img_width = (filters.shape[1] * zoom + margin) * nb_cols - margin

    if not relu:
        highs = []
        lows = []
        for color in range(depth):
            lows.append(np.min(filters[:, :, color, :]))
            highs.append(np.max(filters[:, :, color, :]))

    colored = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    colored[:] = 255
    for i in range(nb_filt):
        f = filters[:, :, :, i]
        x0, y0 = (f.shape[0] * zoom + margin) * (i // nb_cols), (f.shape[1] * zoom + margin) * (i % nb_cols)
        x1, y1 = x0 + f.shape[0] * zoom, y0 + f.shape[1] * zoom
        tile = f.copy()
        if relu:
            tile[np.where(tile < 0)] = 0
            tile /= np.max(tile)
        else:
            for color in range(depth):
                # noinspection PyUnboundLocalVariable
                tile[:, :, color] = (tile[:, :, color] - lows[color]) / (highs[color] - lows[color])
        # noinspection PyTypeChecker
        colored[x0:x1, y0:y1] = np.repeat(np.repeat(tile * 255, zoom, axis=0), zoom, axis=1)
    c_height = colored.shape[0]
    canvas = np.zeros(((c_height + margin) * 4 - margin, colored.shape[1], depth), dtype=np.uint8)
    canvas[0:c_height, :, :] = colored
    for color in range(depth):
        x0 = (color + 1) * (c_height + margin)
        canvas[x0: x0 + c_height, :, color] = colored[:, :, color]
    show(canvas, name='Convolution filters - First layer')
    cv2.waitKey()


def patchwork(manager, x, shape=(20, 20)):
    a, b = shape
    rw, cw = manager.r_width, manager.c_width
    idx = 0
    while True:
        canvas = np.zeros((a * rw, b * cw, manager.depth), dtype=np.uint8)
        for j in range(a):
            for k in range(b):
                if idx < len(x):
                    canvas[j * rw:(j + 1) * rw, k * cw:(k + 1) * cw] = x[idx]
                else:
                    print("Images {} -> {}".format(idx - (j * b + k), idx))
                    yield canvas
                    return
                idx += 1
        print("Images {} -> {}".format(idx - a * b, idx - 1))
        yield canvas


def visualize_inputs(manager, x, shape=(20, 20)):
    for img in patchwork(manager, x, shape):
        show(img, name='Patchwork')
        if chr(cv2.waitKey()) == 'q':
            break


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-f', '--filters', action='store_true', default=False, help="Display convolution filters.")
    parser.add_argument('-p', '--patchwork', help="Show a patchwork of the xs in the matrix file.")
    return parser


if __name__ == '__main__':
    manager = NNManager()
    args = get_argparser().parse_args()

    if args.filters:
        visualize_l0(manager)

    extension = '.npz'
    if args.patchwork is not None:
        if args.patchwork.endswith(extension):
            # visualize_inputs(manager, np.load(args.patchwork)['X'])
            xt, yt, xe, ye = split_data(cvconf.snapshot_dir)
            visualize_inputs(manager, xt)
        else:
            print('Unsupported arg for command -p (--patchwork): expecting \'{}\' files'.format(extension))
