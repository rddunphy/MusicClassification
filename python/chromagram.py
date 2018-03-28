import numpy as np
import matplotlib.pyplot as plt
import operator

from audio_utilities import read_mono

nfft = 2048
step = 128
nbins = 12
A5 = 880
st = 2 ** (1 / float(nbins))

tunechroma1 = [np.log2(A5 * st ** i) for i in range(nbins)]
tunechroma2 = [int(np.log2(A5 * st ** i)) for i in range(nbins)]

chroma = np.asarray(tunechroma1) - np.asarray(tunechroma2)


def run(input_path):
    rate, mono = read_mono(input_path)

    plt.figure(1)
    plt.title('Source wave')
    plt.plot(mono)

    spectrum, freqs, _, _ = plt.specgram(mono, Fs=rate, window=np.hamming(nfft), NFFT=nfft, noverlap=nfft - step, Fc=0)

    freqs = freqs[1:, ]  # ?

    freqschroma = np.asarray(np.log2(freqs)) - np.asarray([int(np.log2(f)) for f in freqs])

    nfreqschroma = len(freqschroma)

    CD = np.zeros((nfreqschroma, nbins))

    for i in range(nbins):
        CD[:, i] = np.abs(freqschroma - chroma[i])

    FlipMatrix = np.flipud(CD)

    min_index = []
    min_value = []

    for i in reversed(range(FlipMatrix.shape[0])):
        index, value = min(enumerate(FlipMatrix[i]), key=operator.itemgetter(1))
        min_index.append(index)
        min_value.append(value)

    # Numpy Array for Chroma Scale population
    CS = np.zeros((len(chroma), spectrum.shape[1]))

    Magnitude = np.log(abs(spectrum[1:, ]))

    for i in range(CS.shape[0]):

        # Find index value in min_index list
        a = [index for index, x in enumerate(min_index) if x == i]

        # Numpy Array for values in each index
        AIndex = np.zeros((len(a), spectrum.shape[1]))

        t = 0
        for value in a:
            AIndex[t, :] = Magnitude[value, :]
            t = t + 1

        MeanMag = []
        for M in AIndex.T:
            MeanMag.append(np.mean(M))

        CS[i, :] = MeanMag

    # normalize the chromagram array
    CS = CS / CS.max()

    plt.figure(2)
    plt.title('Chromagram')
    plt.imshow(CS.astype('float64'), interpolation='nearest', origin='lower', aspect='auto')

    plt.show()


if __name__ == '__main__':
    run('scale.wav')
