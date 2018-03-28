"""DEPRECATED - Old corpus assembly script.

This script was used to generate a corpus of wave files of the entire corpus
in a single directory. This approach was abandoned in favour of the BCRM and
MS datasets, which have fixed length samples.
"""

import contextlib
import csv
import os
import pickle
import uuid
import wave
from shutil import copyfile

from file_converter import convert_to_wav

CORPUS_PATH = '../corpus'
WAVE_PATH = CORPUS_PATH + '/wave'
SOURCE_PATH = CORPUS_PATH + '/audio_sources'
CSV_PATH = CORPUS_PATH + '/corpus.csv'
INDEX_PATH = CORPUS_PATH + '/index.p'
AUDIO_EXTENSIONS = ['mp3', 'wav', 'wma', 'm4a', 'mp4']


def _get_wav_duration(file):
    with contextlib.closing(wave.open(file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def _generate_id():
    return str(uuid.uuid4())


def _build_file_path(file_id):
    return '/'.join([WAVE_PATH, file_id + '.wav'])


def _process_file(file, name, indexed_paths, period, composer):
    file = convert_to_wav(file, mono=True)
    file_id = _generate_id()
    destination = _build_file_path(file_id)
    copyfile(file, destination)
    duration = _get_wav_duration(file)
    indexed_paths.append(file)
    return [file_id, name, duration, period, composer]


def _valid_file_type(file):
    extension = file.split('.')[-1]
    return extension in AUDIO_EXTENSIONS


def _process_comp_dir(root, indexed_paths, period, composer):
    rows = []
    for file in os.listdir(root):
        path = root + '/' + file
        if os.path.isfile(path) and _valid_file_type(path):
            if path not in indexed_paths:
                rows.append(_process_file(path, file, indexed_paths, period, composer))
        else:
            print('    Invalid format: {}'.format(file))
    if rows:
        print('    Found {0} new tracks by {1}.'.format(len(rows), composer))
    return rows


def _process_period_dir(root, indexed_paths, period):
    print('  Processing tracks in {}...'.format(period))
    rows = []
    for file in os.listdir(root):
        path = root + '/' + file
        if os.path.isdir(path):
            rows += _process_comp_dir(path, indexed_paths, period, file)
    return rows


# def _process_set_dir(root, indexed_paths, data_set):
#     print('Processing files in data set {}...'.format(data_set))
#     rows = []
#     for file in os.listdir(root):
#         path = root + '/' + file
#         if os.path.isdir(path):
#             rows += _process_period_dir(path, indexed_paths, data_set, file)
#     return rows


def _load_indexed_paths():
    try:
        return pickle.load(open(INDEX_PATH, 'rb'))
    except (OSError, IOError):
        return []


def _store_indexed_paths(paths):
    pickle.dump(paths, open(INDEX_PATH, 'wb'))


def _process_sources_dir():
    indexed_paths = _load_indexed_paths()
    rows = []
    for file in os.listdir(SOURCE_PATH):
        path = SOURCE_PATH + '/' + file
        if os.path.isdir(path):
            rows += _process_period_dir(path, indexed_paths, file)
    _store_indexed_paths(indexed_paths)
    return rows


def _read_csv():
    if not os.path.isfile(CSV_PATH):
        return []
    with open(CSV_PATH, 'r') as csv_file:
        rows = []
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            rows.append(row)
        return rows


def _write_to_csv(rows):
    with open(CSV_PATH, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(rows)


def run():
    old_rows = _read_csv()
    new_rows = _process_sources_dir()
    _write_to_csv(old_rows + new_rows)
    print('Added {} new tracks.'.format(len(new_rows)))
    print('Total: {} tracks.'.format(len(old_rows) + len(new_rows) - 1))


if __name__ == '__main__':
    run()
