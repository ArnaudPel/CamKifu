import argparse
import os
import re as regex
from os.path import join, isfile, exists

import cv2
import numpy as np

from camkifu.config import cvconf
from camkifu.core.imgutil import show, destroy_win
from camkifu.stone.nn_manager import TRAIN_DAT_MTX, NNManager, PNG_SUFFIX, TRAIN_DAT_SUFFIX

__author__ = 'Arnaud Peloquin'


def merge_npz(train_data_dir, pattern):
    inputs = []
    labels = []
    files = []
    for mat in [f for f in os.listdir(train_data_dir) if f.endswith('.npz')]:
        if regex.match(pattern, mat):
            data = np.load(join(train_data_dir, mat))
            inputs.append(data['X'])
            labels.append(data['Y'])
            files.append(mat)
    x = np.concatenate(inputs, axis=0)
    print('Merged {} inputs -> {}'.format(len(inputs), x.shape))
    y = np.concatenate(labels, axis=0)
    return x, y, files


def display_histo(labels, nb_bins=3 ** 4):
    """
    labels -- the label vectors as fed to the neural network
    """
    histo = raw_histo(labels, nb_bins)
    cv2.normalize(histo, histo, 0, 255, cv2.NORM_MINMAX)
    zoom = 7
    chart_width = nb_bins * zoom
    bins = np.arange(0, chart_width, zoom).reshape(nb_bins, 1)
    pts = np.column_stack((bins, (np.int32(np.around(histo)))))
    canvas = np.zeros((300, chart_width, 3))
    cv2.polylines(canvas, [pts], False, (255, 255, 255))
    win_name = 'Training data labels distribution'
    show(np.flipud(canvas), name=win_name)
    cv2.waitKey()
    destroy_win(win_name)


def raw_histo(labels, nb_bins=3 ** 4):
    y = np.zeros((labels.shape[0], 1), dtype=np.uint8)
    for i, label_vect in enumerate(labels):
        y[i] = np.argmax(label_vect)
    return cv2.calcHist([y], [0], None, [nb_bins], [0, nb_bins])


def distrib_imbalance(labels, nb_bins=3 ** 4):
    """ Return an sorted list mapping each label to its 'balance ratio'
    Ratio < 1 means the label is not frequent enough
    Ratio > 1 means the label is too frequent

    """
    histo = raw_histo(labels, nb_bins=nb_bins)
    distr = histo.astype(np.float32) / np.sum(histo)
    res = [(label, freq[0] * nb_bins) for label, freq in enumerate(distr)]
    return sorted(res, key=lambda t: t[1])


def split_data(base_dir):
    tfile = join(base_dir, TRAIN_DAT_MTX)
    if isfile(tfile):
        tdat = np.load(tfile)
        xt, yt = tdat['Xt'], tdat['Yt']
        xe, ye = tdat['Xe'], tdat['Ye']
        print("Loaded datasets from [{}] {} | {}".format(TRAIN_DAT_MTX, len(xt), len(xe)))
    else:
        x_tot, y_tot, _ = merge_npz(base_dir, '(.*\-train data.*)|(arch\d*)')
        indices = np.indices([x_tot.shape[0]])[0]
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        xt = x_tot[indices[:split_idx], :]
        yt = y_tot[indices[:split_idx], :]
        xe = x_tot[indices[split_idx:], :]
        ye = y_tot[indices[split_idx:], :]
        np.savez(tfile, Xt=xt, Yt=yt, Xe=xe, Ye=ye)
        print("Created NEW datasets: [{}]".format(TRAIN_DAT_MTX))
    return xt, yt, xe, ye


def extract_ys(base_dir):
    """ Extract the Y matrix from every train-data file for which no ref game can be found.

    """
    for mat in [f for f in os.listdir(base_dir) if f.endswith(PNG_SUFFIX)]:
        path = join(base_dir, mat)
        if not isfile(NNManager.get_ref_game(path)):
            dat_path = path.replace(PNG_SUFFIX, TRAIN_DAT_SUFFIX)
            if isfile(dat_path):
                np.savez(NNManager.get_ref_y(path), Y=np.load(dat_path)['Y'])


def archive(idx):
    assert type(idx) is int
    cd = cvconf.snapshot_dir  # current directory
    arch_dir = join(cd, 'arch{}'.format(idx))
    arch_file = arch_dir + '.npz'
    if exists(arch_dir):
        raise IsADirectoryError(arch_dir, "Directory already exists")
    if isfile(arch_file):
        raise FileExistsError(arch_file)
    xa, ya, files = merge_npz(cd, '.*\-train data.*')
    np.savez(arch_file, X=xa, Y=ya)
    os.makedirs(arch_dir)
    for f in files:
        os.rename(join(cd, f), join(arch_dir, f))
        pic = f.replace(TRAIN_DAT_SUFFIX, PNG_SUFFIX)
        os.rename(join(cd, pic), join(arch_dir, pic))


def run_batch(manager, base_dir=cvconf.snapshot_dir, train=True, evaluate=True, epochs=15):
    try:
        x_train, y_train, x_eval, y_eval = split_data(base_dir)
        if train:
            manager.train(x_train, y_train, batch_size=445, nb_epoch=epochs, lr=0.003)
        if evaluate:
            manager.evaluate(x_eval, y_eval)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-t', '--train', action='store_true', default=False, help="Run neural net training.")
    parser.add_argument('-v', '--eval', action='store_true', default=False, help="Evaluate the current neural network.")
    parser.add_argument('-e', '--epoch', default=15, help="Set the number of epochs (applies to training only).")
    parser.add_argument('-a', '--arch', default=0, type=int, help="Archive snapshots and their associated labels")
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    manager = NNManager()

    if args.train or args.eval:
        run_batch(manager, train=args.train, evaluate=args.eval, epochs=args.epoch)

    if args.arch:
        archive(args.arch)

    # xt, yt, xe, ye = split_data(cvconf.snapshot_dir)
    # manager.visualize(xe)

    # x, y, _ = merge_npz(cvconf.snapshot_dir, 'snapshot\-3 \(\d*\).*')
    # manager.evaluate(x, y)

    # x, y, _ = NNManager.merge_npz(cvconf.snapshot_dir, '(.*\-train data.*)|(arch\d*)')
    # distrib = NNManager.distrib_imbalance(y)
    # for label, ratio in distrib:
    #     print("{} : {:.2f}".format(NNManager.compute_stones(label), ratio))

    # extract_ys(cvconf.snapshot_dir)

