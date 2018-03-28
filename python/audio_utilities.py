"""Various utility functions for use in audio scripts."""

from scipy.io import wavfile

AUDIO_EXTENSIONS = ['mp3', 'wav', 'wma', 'm4a', 'mp4']


def read_mono(file):
    """DEPRECATED - use PyDub's set_channels(1) instead."""
    rate, audio = wavfile.read(file)
    mono = [0.5 * x[0] + 0.5 * x[1] for x in audio]
    return rate, mono


def valid_file_type(file):
    """Verify that a file name has the extension of a valid audio file."""
    extension = file.split('.')[-1]
    return extension in AUDIO_EXTENSIONS


def print_progress(progress, indent=0):
    """Print a progress bar. Call print() after completion.

    Args:
        progress (float): between 0 and 1
        indent (int): number of spaces to print before progress bar
    """
    n = round(progress * 20)
    print('\r{0}[{1}{2}] {3}%'.format(' ' * indent, '#' * n, ' ' * (20 - n),
                                      round(progress * 100)), end='\r')
