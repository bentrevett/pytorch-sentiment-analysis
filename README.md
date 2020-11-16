## Note: This repo is currently being updated for the new torchtext API!

**As of November 2020 the new torchtext experimental API - which will be replacing the current API - is in development. To maintain legacy support, the implementations below will not be removed, but will probably be moved to a `legacy` folder at some point. Updated tutorials using the new API are currently being written, though the new API is not finalized so these are subject to change but I will do my best to keep them up to date. The new tutorials are located in the [experimental](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/master/experimental) folder, and require PyTorch 1.7, Python 3.8 and a torchtext built from the master branch - not installed via pip - see the README in the torchtext repo for instructions on how to build torchtext from master.**

**If you have any feedback in regards to them, please submit and issue with the word "experimental" somewhere in the title.**

# PyTorch Sentiment Analysis

This repo contains tutorials covering how to perform sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch) 1.7 and [torchtext](https://github.com/pytorch/text) 0.8 using Python 3.8.

The first 2 tutorials will cover getting started with the de facto approach to sentiment analysis: recurrent neural networks (RNNs). The third notebook covers the [FastText](https://arxiv.org/abs/1607.01759) model and the final covers a [convolutional neural network](https://arxiv.org/abs/1408.5882) (CNN) model.

There are also 2 bonus "appendix" notebooks. The first covers loading your own datasets with TorchText, while the second contains a brief look at the pre-trained word embeddings provided by TorchText.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-sentiment-analysis/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).

To install TorchText:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the English models with:

``` bash
python -m spacy download en
```

For tutorial 6, we'll use the transformers library, which can be installed via:

```bash
pip install transformers
```
These tutorials were created using version 1.2 of the transformers library.

## Tutorials

* 1 - [Simple Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

    This tutorial covers the workflow of a PyTorch with TorchText project. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model will be simple and achieve poor performance, but this will be improved in the subsequent tutorials.

* 2 - [Upgraded Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: using packed padded sequences, loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization.

* 3 - [Faster Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb)

    After we've covered all the fancy upgrades to RNNs, we'll look at a different approach that does not use RNNs. More specifically, we'll implement the model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759). This simple model achieves comparable performance as the *Upgraded Sentiment Analysis*, but trains much faster.

* 4 - [Convolutional Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)

    Next, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

* 5 - [Multi-class Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb)
    
    Then we'll cover the case where we have more than 2 classes, as is common in NLP. We'll be using the CNN model from the previous notebook and a new dataset which has 6 classes.

* 6 - [Transformers for Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb)

    Finally, we'll show how to use the transformers library to load a pre-trained transformer model, specifically the BERT model from [this](https://arxiv.org/abs/1810.04805) paper, and use it to provide the embeddings for text. These embeddings can be fed into any model to predict sentiment, however we use a gated recurrent unit (GRU).

## Appendices

* A - [Using TorchText with your Own Datasets](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb)

    The tutorials use TorchText's built in datasets. This first appendix notebook covers how to load your own datasets using TorchText.

* B - [A Closer Look at Word Embeddings](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/B%20-%20A%20Closer%20Look%20at%20Word%20Embeddings.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/B%20-%20A%20Closer%20Look%20at%20Word%20Embeddings.ipynb)

    This appendix notebook covers a brief look at exploring the pre-trained word embeddings provided by TorchText by using them to look at similar words as well as implementing a basic spelling error corrector based entirely on word embeddings.

* C - [Loading, Saving and Freezing Embeddings](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/C%20-%20Loading%2C%20Saving%20and%20Freezing%20Embeddings.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/C%20-%20Loading%2C%20Saving%20and%20Freezing%20Embeddings.ipynb)

    In this notebook we cover: how to load custom word embeddings, how to freeze and unfreeze word embeddings whilst training our models and how to save our learned embeddings so they can be used in another model.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

- http://anie.me/On-Torchtext/
- http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
- https://github.com/spro/practical-pytorch
- https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
- https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
- https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
- https://github.com/Shawn1993/cnn-text-classification-pytorch
