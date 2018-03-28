#!/usr/bin/env python3

import argparse
import csv
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

# Periods: Baroque, Classical, Romantic, Modern
PERIOD_LABELS = ['b', 'c', 'r', 'm']
# Composers: Corelli, Vivaldi, Bach, Haydn, Mozart, Beethoven,
# Barhms, Tchaikovsky, Mahler, Stravinsky, Shostakovich, Messiaen
COMP_LABELS = ['cor', 'viv', 'bac', 'hay', 'moz', 'bee',
               'bra', 'tch', 'mah', 'str', 'sho', 'mes']


def _one_hot(sample, labels, comp):
    if comp:
        return [1 if sample[2:5] == x else 0 for x in labels]
    return [1 if sample[0] == x else 0 for x in labels]


def _load_data(path, labels, comp, limit=None):
    samples = os.listdir(path)
    np.random.shuffle(samples)
    d = []
    o = []
    if not limit:
        limit = len(samples)
    for i in range(limit):
        sample = samples[i]
        file = os.path.join(path, sample)
        d.append(pickle.load(open(file, 'rb')))
        o.append(_one_hot(sample, labels, comp))
    return d, o


def run(corpus_path, n_hidden, n_epochs, chkpt_interval, train_samples,
        batch_size, sample_length, n_bins, comp):
    n_batches = int(train_samples/batch_size)
    labels = COMP_LABELS if comp else PERIOD_LABELS

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_path = os.path.join("tf", "lstm_data_" + run_id)
    log_path = os.path.join(data_path, "log")
    acc_csv_path = os.path.join(data_path, "acc.csv")

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    with open(acc_csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Epoch", "Training accuracy",
                         "Validation accuracy"])

    valid_path = os.path.join(corpus_path, "valid")
    valid_input, valid_output = _load_data(valid_path, labels, comp)

    data = tf.placeholder(
        tf.float32, [None, sample_length, n_bins])
    target = tf.placeholder(tf.float32, [None, len(labels)])

    cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)

    val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    W = tf.Variable(tf.truncated_normal(
        [n_hidden, int(target.get_shape()[1])]))
    b = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(last, W) + b)

    clipped = tf.clip_by_value(prediction, 1e-10, 1.0)
    cross_entropy = -tf.reduce_sum(target * tf.log(clipped))

    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1),
                            tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
    accuracy = tf.subtract(1.0, error)

    conf_matrix = tf.confusion_matrix(
        tf.argmax(target, 1), tf.argmax(prediction, 1))

    tf.summary.scalar('accuracy', accuracy)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_log_path = os.path.join(log_path, "train")
    train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
    test_log_path = os.path.join(log_path, "validation")
    test_writer = tf.summary.FileWriter(test_log_path)

    saver = tf.train.Saver()

    tf.global_variables_initializer().run(session=sess)

    train_path = os.path.join(corpus_path, "train")
    for e in range(n_epochs):
        train_input, train_output = _load_data(
            train_path, labels, comp, limit=train_samples)
        ptr = 0
        for _ in range(n_batches):
            in_ = train_input[ptr:ptr+batch_size]
            out = train_output[ptr:ptr+batch_size]
            ptr += batch_size
            sess.run(minimize, {data: in_, target: out})

        train_sum, train_acc = sess.run(
            [merged, accuracy],
            {data: train_input, target: train_output})
        train_writer.add_summary(train_sum, e)

        # Calculate and write validation accuracy
        test_sum, test_acc = sess.run(
            [merged, accuracy],
            {data: valid_input, target: valid_output})
        test_writer.add_summary(test_sum, e)
        print(("Epoch {:3d}: training accuracy {:3.1f}%, "
               "validation accuracy {:3.1f}%").format(
                   e + 1, 100 * train_acc, 100 * test_acc))

        # Save accuracy to CSV
        with open(acc_csv_path, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            csv_row = [e+1, 100 * train_acc, 100 * test_acc]
            writer.writerow(csv_row)

        if (e+1) % chkpt_interval == 0 or e == n_epochs - 1:
            # Create checkpoint and print confusion matrix
            chkpt_dir = "chkpt_{}".format(e+1)
            save_path = os.path.join(data_path, chkpt_dir)
            saver.save(sess, save_path)
            print(sess.run(conf_matrix,
                           {data: valid_input, target: valid_output}))

    # Print final test accuracy and confusion matrix
    test_input, test_output = _load_data(
        corpus_path + "/test", labels, comp)
    test_acc = sess.run(accuracy,
                        {data: test_input, target: test_output})
    print("Final test accuracy {:3.1f}%".format(100 * test_acc))
    print(sess.run(conf_matrix,
                   {data: test_input, target: test_output}))

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train an LSTM to classify HPCP samples.")
    parser.add_argument(
        '--corpus_path',
        type=str,
        default="corpus/bcrm_hpcp_12bin",
        help="HPCP sample directory path")
    parser.add_argument(
        '--n_hidden',
        type=int,
        default=24,
        help="Number of hidden layers")
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1000,
        help="Number of epochs in training run")
    parser.add_argument(
        '--chkpt_interval',
        type=int,
        default=50,
        help="Number of epochs between checkpoints")
    parser.add_argument(
        '--train_samples',
        type=int,
        default=2000,
        help="Number of samples used in each training step")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help="Number of samples in each minibatch used for SGD")
    parser.add_argument(
        '--sample_length',
        type=int,
        default="2570",
        help="Number of HPCP vectors in each sample (use 10322 for 60s)")
    parser.add_argument(
        '--n_bins',
        type=int,
        default=12,
        help="Number of HPCP bins")
    parser.add_argument(
        '--comp',
        action='store_true',
        help="Classify by composer instead of by period")
    args = parser.parse_args()
    run(args.corpus_path, args.n_hidden, args.n_epochs,
        args.chkpt_interval, args.train_samples, args.batch_size,
        args.sample_length, args.n_bins, args.comp)
