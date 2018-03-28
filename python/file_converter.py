import argparse
import os
from pydub import AudioSegment


def convert_to_wav(file, mono=False):
    if file.endswith('.wav') and not mono:
        return file
    audio = AudioSegment.from_file(file)
    if mono:
        audio = audio.set_channels(1)
    converted = file[:file.rfind('.')] + '.wav'
    audio.export(converted, format='wav')
    os.remove(file)
    return converted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert file to .wav format.')
    parser.add_argument('file', help='Source audio file')
    args = parser.parse_args()
    convert_to_wav(args.file)
