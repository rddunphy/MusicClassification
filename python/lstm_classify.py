#!/usr/bin/env python3
r"""Script to classify a sample either by period or composer."""

import argparse
import os
import pickle
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PERIOD_GRAPH_PATH = "tf/lstm/period_graph.pb"
COMP_GRAPH_PATH = "tf/lstm/comp_graph.pb"
SAMPLE_PATH = "corpus/demo/hpcp/"

PERIOD_LABELS = ['b', 'c', 'r', 'm']
PERIOD_OUTPUT_LABELS = ["Baroque", "Classical", "Romantic", "Modern"]

COMP_LABELS = [
    'cor', 'viv', 'bac', 'hay', 'moz', 'bee', 'bra', 'tch', 'mah', 'str',
    'sho', 'mes'
]
COMP_OUTPUT_LABELS = [
    "Corelli", "Vivaldi", "Bach", "Haydn", "Mozart", "Beethoven", "Brahms",
    "Tchaikovsky", "Mahler", "Stravinsky", "Shostakovich", "Messiaen"
]


def _load_graph(frozen_graph_path):
    with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
            return graph


def _one_hot(sample, comp):
    if comp:
        return [1 if sample[2:5] == x else 0 for x in COMP_LABELS]
    return [1 if sample[0] == x else 0 for x in PERIOD_LABELS]


def _load_data(path, comp):
    d = [pickle.load(open(path, 'rb'))]
    o = [_one_hot(os.path.basename(path), comp)]
    return d, o


def _get_label(index, comp):
    if comp:
        return COMP_OUTPUT_LABELS[index]
    return PERIOD_OUTPUT_LABELS[index]


def _print_prediction(period_graph_path, comp_graph_path, sample_path):
    labels = []
    predictions = []
    if period_graph_path:
        pl, pp = prediction(period_graph_path, sample_path, False)
        labels += [pl]
        predictions += [pp]
    if comp_graph_path:
        cl, cp = prediction(comp_graph_path, sample_path, True)
        labels += [cl]
        predictions += [cp]
    print("Correct Label: {}".format("/".join(labels)))
    print("Prediction:    {}".format("/".join(predictions)))


def prediction(graph_path, sample_path, comp):
    """Get actual label and prediction for an HPCP samples

    returns:
        (label, prediction)
    """
    graph = _load_graph(graph_path)
    data = graph.get_tensor_by_name('prefix/Placeholder:0')
    one_hot = graph.get_tensor_by_name('prefix/Placeholder_1:0')
    label = graph.get_tensor_by_name('prefix/ArgMax:0')
    pred = graph.get_tensor_by_name('prefix/ArgMax_1:0')
    sample = _load_data(sample_path, comp)
    with tf.Session(graph=graph) as sess:
        inputs = {data: sample[0], one_hot: sample[1]}
        out = sess.run([label, pred], feed_dict=inputs)
        label_str = _get_label(out[0][0], comp)
        pred_str = _get_label(out[1][0], comp)
        return label_str, pred_str


if __name__ == '__main__':
    period_default_graph_path = os.path.join(ROOT_DIR, PERIOD_GRAPH_PATH)
    comp_default_graph_path = os.path.join(ROOT_DIR, COMP_GRAPH_PATH)
    default_sample_dir = os.path.join(ROOT_DIR, SAMPLE_PATH)
    parser = argparse.ArgumentParser(
        description="Classify an HPCP sample by composer")
    parser.add_argument(
        'sample',
        type=str,
        help="the ID of the sample file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--comp-only',
        action='store_true',
        help="print only composer prediction")
    group.add_argument(
        '--period-only',
        action='store_true',
        help="print only period prediction")
    parser.add_argument(
        '--comp-graph-path',
        type=str,
        default=comp_default_graph_path,
        help="the path of the graph used to predict the composer")
    parser.add_argument(
        '--period-graph-path',
        type=str,
        default=period_default_graph_path,
        help="the path of the graph used to predict the period")
    parser.add_argument(
        '--sample-dir',
        type=str,
        default=default_sample_dir,
        help="directory path containing HPCP samples")
    args = parser.parse_args()
    _sp = os.path.join(args.sample_dir, "{}.p".format(args.sample))
    if args.comp_only:
        args.period_graph_path = None
    if args.period_only:
        args.comp_graph_path = None
    _print_prediction(args.period_graph_path, args.comp_graph_path, _sp)
