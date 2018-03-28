import argparse

import numpy
from matplotlib import pyplot
from numpy import fft

from audio_utilities import read_mono


def run(file):
    rate, mono = read_mono(file)

    time = numpy.arange(0, float(mono.shape[0]), 1) / rate
    pyplot.subplot(311)
    pyplot.plot(time, mono, linewidth=0.02, alpha=0.8)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Amplitude')

    fourier = fft.fft(mono)
    n = len(mono)
    fourier = fourier[: int(n/2)]
    fourier = fourier / float(n)
    freq = numpy.arange(0, (n / 2), 1.0) * (rate * 1.0 / n)
    pyplot.subplot(312)
    pyplot.plot(freq / 1000, 10 * numpy.log10(fourier), linewidth=0.02)
    pyplot.xlabel('Frequency (kHz)')
    pyplot.ylabel('Power (dB)')

    pyplot.subplot(313)
    pyplot.specgram(mono, Fs=rate, NFFT=1024)
    pyplot.xlabel('Time (s)')
    pyplot.ylabel('Frequency (Hz)')
    pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display graphs of input audio file.')
    parser.add_argument('file', help='source audio file (must be .wav)')
    args = parser.parse_args()
    run(args.file)
