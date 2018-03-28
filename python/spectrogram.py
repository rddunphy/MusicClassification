import argparse

import numpy as np
from matplotlib import pyplot
from numpy import fft
from scipy import signal
from scipy.io import wavfile


def run(file):
    fr, audio = wavfile.read(file)
    mono = [0.5 * x[0] + 0.5 * x[1] for x in audio]
    pyplot.specgram(mono, Fs=fr, NFFT=1024)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Frequency (Hz)')
    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display graphs of input audio file.')
    parser.add_argument('file', help='source audio file (must be .wav)')
    args = parser.parse_args()
    run(args.file)
