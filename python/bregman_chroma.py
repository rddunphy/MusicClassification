#!/usr/bin/env python2

from bregman.suite import Chromagram, wavread


def show_chromagram(audio_file):
    audio, fs, _ = wavread(audio_file)
    chroma = Chromagram(audio, sample_rate=fs, nfft=16384, wfft=8192,
                        nhop=2205)
    chroma.feature_plot(dbscale=False, normalize=True)


if __name__ == '__main__':
    show_chromagram("beethoven.wav")
