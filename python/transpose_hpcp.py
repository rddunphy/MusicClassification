import os
import pickle

n_transpositions = 5


def _load(src_dir, file):
    path = os.path.join(src_dir, file)
    return pickle.load(open(path, 'rb'))


def _save(data, out_dir, file, t):
    # Add transposition interval to file name
    file = file.split('.')[0]
    file += "+{:02d}.p".format(t)
    path = os.path.join(out_dir, file)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def transpose(src_dir, out_dir, file):
    data = _load(src_dir, file)
    # Save unchanged for null transposition
    _save(data, out_dir, file, 0)
    for i in range(n_transpositions):
        for r, row in enumerate(data):
            # Rotate list by one
            data[r] = [row[-1]] + row[:-1]
        _save(data, out_dir, file, i+1)


def transpose_files(src_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for file in os.listdir(src_dir):
        path = os.path.join(src_dir, file)
        if os.path.isdir(path):
            transpose_files(path, out_dir)
        else:
            transpose(src_dir, out_dir, file)


if __name__ == '__main__':
    transpose_files("corpus/bcrm_hpcp_12bin", "corpus/bcrm_hpcp_12bin_transposed")
