"""Split audio files in a directory into segments of a given length."""

import argparse
import csv
import os
import uuid
from timeit import default_timer as timer

from pydub import AudioSegment

from audio_utilities import valid_file_type


def _write_index(rows, path):
    if path:
        print('Saving index to {}...'.format(path))
        with open(path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['File ID', 'Source path', 'No. segments'])
            writer.writerows(rows)


def _save_segment(segment, output_path, fade):
    segment = segment.fade_in(fade).fade_out(fade)
    segment.export(output_path, format='wav')


def _print_progress(progress):
    n = round(progress / 5)
    print('\r[{0}{1}] {2}%'.format('#' * n, ' ' * (20 - n), round(progress)),
          end='\r')


def _generate_output_path(output_dir, file_id, n_files, separator):
    output_path = '{0}/{1}{2}{3:05d}.wav'.format(output_dir, file_id,
                                                 separator, n_files)
    return output_path


def _load_audio(audio_file, sample_rate, stereo):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(sample_rate)
    if stereo:
        audio = audio.set_channels(2)
    else:
        audio = audio.set_channels(1)
    return audio


def _process_file(file, out_dir, sample_rate, stereo, seg_length, overlap,
                  fade, separator):
    n_output_files = 0
    file_id = str(uuid.uuid4())
    print(file)
    _print_progress(0)
    remaining = _load_audio(file, sample_rate, stereo)
    full_length = len(remaining)
    offset = seg_length - overlap
    while len(remaining) > seg_length:
        segment = remaining[:seg_length]
        remaining = remaining[offset:]
        output_path = _generate_output_path(out_dir, file_id, n_output_files,
                                            separator)
        _save_segment(segment, output_path, fade)
        n_output_files += 1
        _print_progress(100 - (100 * len(remaining) / full_length))
    _print_progress(100)
    print()
    return [file_id, file, n_output_files]


def _process_dir(src_dir, out_dir, sample_rate, stereo, seg_length, overlap,
                 fade, separator, recursive):
    csv_rows = []
    for file in os.listdir(src_dir):
        file_path = '/'.join([src_dir, file])
        if os.path.isfile(file_path) and valid_file_type(file_path):
            try:
                r = _process_file(file_path, out_dir, sample_rate, stereo,
                                  seg_length, overlap, fade, separator)
                csv_rows.append(r)
            except OSError:
                print('Problem encountered while segmenting {}.'.format(
                    file_path))
        elif os.path.isdir(file_path) and recursive:
            csv_rows += _process_dir(file_path, out_dir, sample_rate, stereo,
                                     seg_length, overlap, fade, separator,
                                     recursive)
    return csv_rows


def _tidy_path(path):
    if path.endswith('/'):
        path = path[:-1]
    return path


def _resolve_csv_path(path, out_dir):
    path = path.replace('{out}', out_dir)
    if not path.endswith('.csv') and path != '':
        return _tidy_path(path) + '/index.csv'
    return path


def _count_output_files(csv_rows):
    return sum([r[2] for r in csv_rows])


def segment_files(src_dir, out_dir, csv_path='', sample_rate=44100,
                  stereo=False, seg_length=10000, overlap=0, fade=250,
                  separator='_', recursive=False):
    """Segment files in source directory and save samples in output directory.

    Args:
        src_dir (str): source directory path
        out_dir (str): output directory path
        csv_path (str): path to save index CSV file to.
        sample_rate (int): sample rate of outputs
        stereo (bool): whether to save output in stereo
        seg_length (int): duration of samples in ms
        overlap (int): overlap between samples in ms
        fade (int): fade in/out duration in ms
        separator (str): string used to separate source file id from sample
            number in output file names
        recursive (bool): whether to recursively process files in
            subdirectories
    """
    if overlap >= seg_length:
        raise Exception("Overlap must be less than segment length.")
    src_dir = _tidy_path(src_dir)
    out_dir = _tidy_path(out_dir)
    csv_path = _resolve_csv_path(csv_path, out_dir)
    start = timer()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_rows = _process_dir(src_dir, out_dir, sample_rate, stereo, seg_length,
                            overlap, fade, separator, recursive)
    end = timer()
    print('Successfully processed {0} files in {1} seconds.'.format(
        len(csv_rows), round(end - start)))
    print('Created {0} samples.'.format(_count_output_files(csv_rows)))
    _write_index(csv_rows, csv_path)
    print('Done.')


def segment_file(audio_file, sample_rate=44100, stereo=False, seg_length=10000,
                 overlap=0, fade=250):
    """Segment audio file and return list of samples.

    Returns:
        A list of Pydub AudioSegment objects corresponding to the specified
        sample parameters.

    Args:
        audio_file (str): source audio file path
        sample_rate (int): sample rate of outputs
        stereo (bool): whether to return samples in stereo
        seg_length (int): duration of samples in ms
        overlap (int): overlap between samples in ms
        fade (int): fade in/out duration in ms
    """
    if overlap >= seg_length:
        raise Exception("Overlap must be less than segment length.")
    remaining = _load_audio(audio_file, sample_rate, stereo)
    samples = []
    offset = seg_length - overlap
    while len(remaining) > seg_length:
        segment = remaining[:seg_length]
        remaining = remaining[offset:]
        segment = segment.fade_in(fade).fade_out(fade)
        samples.append(segment)
    return samples


def _positive(value):
    ival = int(value)
    if ival <= 0:
        raise argparse.ArgumentTypeError(
            '{} is invalid for positive integer.'.format(value))
    return ival


def _nonnegative(value):
    ival = int(value)
    if ival < 0:
        raise argparse.ArgumentTypeError(
            '{} is invalid for non-negative integer.'.format(value))
    return ival


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Segment audio files in a directory into shorter samples.'
    )
    parser.add_argument('src', help='source directory')
    parser.add_argument('out', help='output directory')
    parser.add_argument(
        '--recursive', '-r', action='store_true',
        help='segment all audio files in subdirectories'
    )
    parser.add_argument(
        '--separator', default='_',
        help='string used to separate id from sample number ("_" by default)'
    )
    parser.add_argument(
        '--segment_length', '-l', type=_positive, default=10,
        help='segment length in seconds (10 by default)'
    )
    parser.add_argument(
        '--overlap', '-o', type=_nonnegative, default=0,
        help='overlap between successive segments in seconds (0 by default)'
    )
    parser.add_argument(
        '--fade_length', '-f', type=_nonnegative, default=250,
        help='fade duration in milliseconds (250 by default)'
    )
    parser.add_argument(
        '--sample_rate', '-s', type=_positive, default=44100,
        help='output sample rate in Hz (44100 by default)'
    )
    parser.add_argument(
        '--stereo', action='store_true',
        help='store output as stereo files (mono by default)'
    )
    parser.add_argument(
        '--csv_path', '-c', default='{out}/index.csv',
        help='path to save index CSV ({out}/index.csv by default)'
    )
    args = parser.parse_args()
    segment_files(args.src, args.out, csv_path=args.csv_path,
                  sample_rate=args.sample_rate, stereo=args.stereo,
                  seg_length=args.segment_length * 1000,
                  overlap=args.overlap * 1000, fade=args.fade_length,
                  separator=args.separator, recursive=args.recursive)
