#!/usr/bin/env python3
r"""Classify all samples in a directory and put predictions in a CSV file."""

import argparse
import csv
import os

from lstm_classify import prediction

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PERIOD_GRAPH_PATH = "tf/lstm/period_graph.pb"
COMP_GRAPH_PATH = "tf/lstm/comp_graph.pb"


def _get_index_row(index_path, sample_file):
    sample_id = sample_file.split('.')[0]
    with open(index_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2].split('.')[0] == sample_id:
                r = []
                r.append(row[2].split(".")[0])
                r.append(row[3])
                r.append(row[4])
                r.append(row[0].title())
                r.append(row[1].title())
                return r
    return None


def _create_predictions_index(path):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        headers = ["ID", "Source file", "Time", "Period", "Composer",
                   "Dataset", "Period pred.", "Composer pred."]
        writer.writerow(headers)


def _write_index_row(pred_path, row):
    if not os.path.isfile(pred_path):
        _create_predictions_index(pred_path)
    with open(pred_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _main(src_dir, index_path, dataset):
    period_graph = os.path.join(ROOT_DIR, PERIOD_GRAPH_PATH)
    comp_graph = os.path.join(ROOT_DIR, COMP_GRAPH_PATH)
    pred_path = os.path.join(os.path.dirname(index_path), "lstm_predictions.csv")
    for sample_file in os.listdir(src_dir):
        row = _get_index_row(index_path, sample_file)
        if row:
            sample_dir = os.path.join(src_dir, sample_file)
            _, period_pred = prediction(period_graph, sample_dir, False)
            _, comp_pred = prediction(comp_graph, sample_dir, True)
            entries = [dataset, period_pred, comp_pred]
            _write_index_row(pred_path, row + entries)
        else:
            print("No index row for sample {}...".format(sample_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Classify all samples in a directory."
    )
    parser.add_argument(
        'src_dir',
        type=str,
        help="directory containing pickled HPCP samples"
    )
    parser.add_argument(
        'index_path',
        type=str,
        help="index CSV path"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="",
        help="dataset name"
    )
    args = parser.parse_args()
    _main(args.src_dir, args.index_path, args.dataset)
