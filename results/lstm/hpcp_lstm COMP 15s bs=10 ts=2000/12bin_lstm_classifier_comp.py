#!/usr/bin/env python3

import csv
import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf

HPCP_PATH = "corpus/bcrm_hpcp_12bin"  # Source directory
CHKPT_INTERVAL = 50  # How often to create checkpoints
INPUT_DIMENSION = 12  # Number of HPCP bins
SEQUENCE_LENGTH = 2570  # 10322 for 60 second samples
BATCH_SIZE = 10
LABELS = ['cor', 'viv', 'bac', 'hay', 'moz', 'bee',
          'bra', 'tch', 'mah', 'str', 'sho', 'mes']
N_EPOCHS = 1000
TRAIN_SAMPLES = 2000  # 500 for 60 second samples
N_HIDDEN = 24


def _one_hot(sample):
    return [1 if sample[2:5] == x else 0 for x in LABELS]


def _load_data(path, limit=None):
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
        o.append(_one_hot(sample))
    return d, o


if __name__ == '__main__':
    n_batches = int(TRAIN_SAMPLES/BATCH_SIZE)

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

    valid_path = os.path.join(HPCP_PATH, "valid")
    valid_input, valid_output = _load_data(valid_path)

    data = tf.placeholder(
        tf.float32, [None, SEQUENCE_LENGTH, INPUT_DIMENSION])
    target = tf.placeholder(tf.float32, [None, len(LABELS)])

    cell = tf.nn.rnn_cell.LSTMCell(N_HIDDEN, state_is_tuple=True)

    val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    W = tf.Variable(tf.truncated_normal(
        [N_HIDDEN, int(target.get_shape()[1])]))
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

    train_path = os.path.join(HPCP_PATH, "train")
    for e in range(N_EPOCHS):
        train_input, train_output = _load_data(
            train_path, limit=TRAIN_SAMPLES)
        ptr = 0
        for j in range(n_batches):
            in_ = train_input[ptr:ptr+BATCH_SIZE]
            out = train_output[ptr:ptr+BATCH_SIZE]
            ptr += BATCH_SIZE
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

        if (e+1) % CHKPT_INTERVAL == 0 or e == N_EPOCHS - 1:
            # Create checkpoint and print confusion matrix
            chkpt_dir = "chkpt_{}".format(e+1)
            save_path = os.path.join(data_path, chkpt_dir)
            saver.save(sess, save_path)
            print(sess.run(conf_matrix,
                           {data: valid_input, target: valid_output}))

    # Print final test accuracy and confusion matrix
    test_input, test_output = _load_data(HPCP_PATH + "/test")
    test_acc = sess.run(accuracy,
                        {data: test_input, target: test_output})
    print("Final test accuracy {:3.1f}%".format(100 * test_acc))
    print(sess.run(conf_matrix,
                   {data: test_input, target: test_output}))

    sess.close()
