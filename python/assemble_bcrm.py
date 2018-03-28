"""Builds a reduced corpus from the audio_sources directory."""

import argparse
import csv
import os
import shutil

import numpy as np

from preprocessing.audio_utilities import print_progress
from preprocessing.segment_audio import segment_file

_SAMPLES_PER_COMPOSER = 800
_SAMPLE_LENGTH = 15  # in seconds

_TMP_PATH = "/tmp/bcrm"

_COMPOSERS = {
    'baroque': ['corelli', 'vivaldi', 'bach'],
    'classical': ['haydn', 'mozart', 'beethoven'],
    'romantic': ['brahms', 'tchaikovsky', 'mahler'],
    'modern': ['stravinsky', 'shostakovich', 'messiaen']
}


def _segment_to_tmp_dir(src_path, period, composer):
    csv_rows = []
    if os.path.exists(_TMP_PATH):
        shutil.rmtree(_TMP_PATH)
    os.makedirs(_TMP_PATH)
    files = os.listdir(src_path)
    print_progress(0)
    src_count, out_count = 0, 0
    for src_file_name in files:
        src_file_path = os.path.join(src_path, src_file_name)
        segments = segment_file(src_file_path, seg_length=(_SAMPLE_LENGTH * 1000))
        seg_count = 0
        for segment in segments:
            out_file_name = str(out_count) + ".wav"
            out_file_path = os.path.join(_TMP_PATH, out_file_name)
            csv_rows.append([period, composer, out_file_name, src_file_name,
                             seg_count])
            segment.export(out_file_path, format='wav')
            out_count += 1
            seg_count += 1
        src_count += 1
        print_progress(src_count / len(files))
    print()  # New line
    return csv_rows


def _create_composer_dataset(corpus_root, period, composer):
    rel_path = os.path.join(period, composer)
    src_path = os.path.join(corpus_root, "audio_sources", rel_path)
    out_path = os.path.join(corpus_root, "bcrm", rel_path)
    csv_rows = _segment_to_tmp_dir(src_path, period, composer)
    count = len(csv_rows)
    choices = np.random.choice(count, _SAMPLES_PER_COMPOSER, replace=False)
    reduced_csv_rows = []
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i, choice in enumerate(choices):
        out_file_name = "{0}_{1}_{2:03d}.wav".format(
            period[0], composer[0:3], i)
        out_file_path = os.path.join(out_path, out_file_name)
        shutil.copyfile(os.path.join(_TMP_PATH, str(choice) + ".wav"),
                        out_file_path)
        row = csv_rows[choice]
        row[2] = out_file_name
        reduced_csv_rows.append(row)
    shutil.rmtree(_TMP_PATH)
    return reduced_csv_rows


def _main(corpus_root):
    csv_rows = []
    for period in _COMPOSERS:
        print("Building {}...".format(period))
        for composer in _COMPOSERS[period]:
            print("  {}...".format(composer))
            new_rows = _create_composer_dataset(corpus_root, period, composer)
            csv_rows += new_rows
    print("Writing index...")
    csv_path = os.path.join(corpus_root, "bcrm", "bcrm_index.csv")
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Period', 'Composer', 'File', 'Source', 'Time'])
        writer.writerows(csv_rows)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build reduced corpus from source files."
    )
    parser.add_argument('dir', help='Corpus root directory')
    args = parser.parse_args()
    if args.dir.endswith('/'):
        args.dir = args.dir[:-1]
    _main(args.dir)
