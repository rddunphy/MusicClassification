"""Randomly splits the contents of a directory into training and test sets."""

import argparse
import math
import os
import shutil

import numpy as np


def _move_files(dir_, files, sub_dir):
    if not files:
        return
    dst = os.path.join(dir_, sub_dir)
    if not os.path.exists(dst):
        os.makedirs(dst)
    for file in files:
        shutil.move(os.path.join(dir_, file), os.path.join(dst, file))


def _split(dir_, files, train, test, validation):
    n_files = len(files)
    total = train + test + validation
    n_test = math.ceil(test * n_files / total)
    n_valid = math.ceil(validation * n_files / total)
    n_train = n_files - n_test - n_valid
    np.random.shuffle(files)
    _move_files(dir_, files[:n_train], "train")
    _move_files(dir_, files[n_train:n_train + n_valid], "valid")
    _move_files(dir_, files[n_train + n_valid:], "test")
    return (n_train, n_valid, n_test)


def _form_bcrm_dict(files):
    # Returns a dictionary where each entry corresponds to a composer,
    # and the values are lists of sample files by those composers.
    file_dict = {}
    for f in files:
        key = f[:5]
        if key in file_dict:
            file_dict[key].append(f)
        else:
            file_dict[key] = [f]
    return file_dict


def split_bcrm(dir_, train, validation, test):
    """Split the contents of BCRM directory into train and test sets.

    Preserves the ratio of samples by the same composers.
    Args:
        dir_ (string): The BCRM root directory
        train (int), validation (int), test (int): The ratio of these
            parameters specifies the percentage of samples to go in
            each dataset (e.g. inputs of 3, 1, and 1 result in sets
            of 60%, 20%, and 20% of samples, respectively)
    """
    files = os.listdir(dir_)
    print("Splitting {} files...".format(len(files)))
    file_dict = _form_bcrm_dict(files)
    results = (0, 0, 0)
    for _, comp_files in file_dict.items():
        r = _split(dir_, comp_files, train, test, validation)
        results = tuple(sum(x) for x in zip(results, r))
    print("train={}, valid={}, test={}".format(*results))


def split(dir_, train, validation, test):
    """Split the contents of a directory into train and test sets."""
    files = os.listdir(dir_)
    print("Splitting {} files...".format(len(files)))
    results = _split(dir_, files, train, test, validation)
    print("train={}, valid={}, test={}".format(*results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Split contents of a directory into training, test, and "
                     "validation sets.")
    )
    parser.add_argument('dir', help="Directory to split")
    parser.add_argument('--train', type=int, default=80,
                        help="Proportion of training files")
    parser.add_argument('--test', type=int, default=10,
                        help="Proportion of test files")
    parser.add_argument('--valid', type=int, default=10,
                        help="Proportion of validation files")
    parser.add_argument('--bcrm', action='store_true')
    args = parser.parse_args()
    if args.bcrm:
        split_bcrm(args.dir, args.train, args.valid, args.test)
    else:
        split(args.dir, args.train, args.valid, args.test)
