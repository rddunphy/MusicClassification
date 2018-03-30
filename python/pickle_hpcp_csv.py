"""Simple script for storing HPCP data from a CSV file as a pickle file."""

import argparse
import csv
import os
import pickle

from audio_utilities import print_progress


def pickle_file(src, out):
    """Store information from HPCP CSV file as a pickle object.

    Args:
        src (str): the CSV files
        out (str): the destination for the pickle files
    """
    all_data = []
    with open(src, 'r') as f:
        cr = csv.reader(f, delimiter=',', quotechar='"')
        for row in cr:
            all_data.append([float(x) for x in row[1:]])
    with open(out, 'wb') as f:
        pickle.dump(all_data, f)


def pickle_files(src_dir, out_dir):
    """Pickle all vamp files in the source directory.

    Args:
        src_dir (str): the directory to process
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    files = os.listdir(src_dir)
    print_progress(0)
    n = 0
    for file in files:
        if file.endswith(".csv"):
            f_name = file.split("_vamp_")[0] + ".p"
            src = os.path.join(src_dir, file)
            out = os.path.join(out_dir, f_name)
            pickle_file(src, out)
        n += 1
        print_progress(n / len(files))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Pickle HPCP CSV vamp files."
    )
    parser.add_argument('src', help="Source directory")
    parser.add_argument('out', help="Output directory")
    args = parser.parse_args()
    pickle_files(args.src, args.out)
