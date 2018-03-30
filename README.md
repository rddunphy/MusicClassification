# Applying machine learning techniques to the classification of classical orchestral music

This  repository contains source code and results for my BEng project at the University of Strathclyde. The final report can be found at `docs/report.pdf`.

## Abstract

Music information retrieval is a growing subfield of machine learning, within which music genre classification is one of the most fundamental tasks. Due to the subjective nature of genres, music classification is a non-trivial task that has been solved only imperfectly to date.

This project investigates two neural network approaches to this problem with a view to classifying classical orchestral music into the four genres of Baroque, Classical, Romantic, and Modern, which are informally considered the major subdivisions of the past four centuries of music history.

A dataset comprised of 40 h of audio samples was assembled. This dataset employs a hierarchical classification, with the four main categories subdivided into three composers each. 

Using convolutional neural networks on spectrograms of this data, accuracies of around 92% were obtained for a binary classification. However, this approach was too memory-intensive to be extended to the full corpus.

The alternative method was the use of a simple recurrent neural network trained on chroma features, which circumvented this difficulty. Under this paradigm, accuracies of 55% were achieved on a four-way classification of the full corpus. This algorithm was also trained to identify composers, yielding a 31% accuracy for a twelve-way classification.

Particular attention was paid to the effects of varying parameters of input data as well as hyper-parameters of the neural networks. It was found that transposition of features was an effective technique for reducing overfitting by artificially increasing the size of the dataset.

## Repository contents

* `corpus`: Details of the tracks included in the corpus and the datasets, as well as some analytical information about the breakdown of these contents.
* `docs`: Project documentation and final report.
* `python`: Original code created for this project. All scripts are written in Python 3.6 unless otherwise specified in the file.
* `results`: Graphs and tabular data summarising the results of training runs.

## Dependencies

* [Numpy/Scipy](https://scipy.org/install.html)
* [TensorFlow](https://www.tensorflow.org/install/)
* [Pydub](https://github.com/jiaaro/pydub)
* [Bregman](http://digitalmusics.dartmouth.edu/~mcasey/bregman/) (used only in `bregman_chroma.py`)

## External resources

The code used as the basis for the CNN/spectrogram classification approach can be found [here](https://www.tensorflow.org/tutorials/audio_recognition).
