import argparse
import math

import cv2
import numpy as np
from theano.tensor.nnet import conv2d, theano, T

from camkifu.core.imgutil import show, destroy_win
from camkifu.stone.nn_manager import NNManager


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


def convolve_l0(manager, img, relu=True):
    x = T.tensor4(name='input')
    filters = manager.get_net().get_weights()[0]
    fs = filters.shape
    w = theano.shared(filters.transpose(3, 2, 0, 1).reshape(fs[3], fs[2], fs[0], fs[1]), name='W')
    f = theano.function([x], conv2d(x, w))
    inputs = manager.generate_xs(img)
    in_s = inputs.shape
    inputs = inputs.transpose(0, 3, 1, 2).reshape(in_s[0], in_s[3], in_s[1], in_s[2])
    out = f(inputs)
    ou_s = out.shape
    out = out.transpose(1, 0, 2, 3).reshape(ou_s[1], ou_s[0], ou_s[2], ou_s[3])
    rshape = (10, 10)
    show(img, name='Base image')
    margin = (in_s[2] - ou_s[2]) // 2
    for idx, img_out in enumerate(out):
        if relu:
            img_out[np.where(img_out < 0)] = 0
        img_out /= np.max(img_out)
        img_out *= 255
        canvas = np.zeros(img.shape[0:2], dtype=np.uint8)
        canvas[:] = 255
        for r in range(rshape[0]):
            rs = r * ou_s[2] + (r+1) * margin
            re = rs + ou_s[2]
            for c in range(rshape[1]):
                cs = c * ou_s[3] + (c+1) * margin
                ce = cs + ou_s[3]
                subidx = r * rshape[0] + c
                canvas[rs:re, cs:ce] = img_out[subidx]
        win_name = 'Convolved by filter #{:d}'.format(idx + 1)
        show(canvas, name=win_name, offset=(img.shape[0]+10, 0))
        if chr(cv2.waitKey()) == 'q':
            break
        destroy_win(win_name)


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


def print_weights(manager):
    print('Nb weights = {0:,}'.format(manager.get_nb_weights()).replace(',', ' '))
    for layer in manager.get_net().get_weights():
        print('\t{}'.format(layer.shape))


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-f', '--filters', action='store_true', default=False, help="Display convolution filters.")
    parser.add_argument('-p', '--patchwork', metavar='NPZ_PATH', help="Show a patchwork of the xs in the matrix file.")
    parser.add_argument('-c', '--convolve', metavar='IMG_PATH', help="Apply 1st convolution layer of the neural net to image.")
    parser.add_argument('-w', '--weights', action='store_true', help="Print basic info on the current model weights.")
    return parser


if __name__ == '__main__':
    manager = NNManager()
    args = get_argparser().parse_args()

    if args.filters:
        visualize_l0(manager)

    if args.patchwork:
        extension = '.npz'
        if args.patchwork.endswith(extension):
            visualize_inputs(manager, np.load(args.patchwork)['X'])
        else:
            print('Unsupported arg for command -p (--patchwork): expecting \'{}\' files'.format(extension))

    if args.convolve:
        extension = ('.png', '.jpg', '.jpeg')
        if args.convolve.lower().endswith(extension):
            convolve_l0(manager, cv2.imread(args.convolve))
        else:
            print('Unsupported arg for command -c (--convolve): expecting \'{}\' files'.format(extension))
    if args.weights:
        print_weights(manager)
