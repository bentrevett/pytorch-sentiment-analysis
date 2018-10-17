# PyTorch Sentiment Analysis

This repo contains tutorials covering how to do sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch) 0.4 and [TorchText](https://github.com/pytorch/text) 0.3 using Python 3.6.

The first 2 tutorials will cover getting started with the de facto approach to sentiment analysis: recurrent neural networks (RNNs). The third notebook covers the [FastText](https://arxiv.org/abs/1607.01759) model and the final covers a [convolutional neural network](https://arxiv.org/abs/1408.5882) (CNN) model.

There are also 2 bonus "appendix" notebooks. The first covers loading your own datasets with TorchText, while the second contains a brief look at the pre-trained word embeddings provided by TorchText.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-sentiment-analysis/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the English models with:

``` bash
python -m spacy download en
```

## Tutorials

* 1 - [Simple Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

    This tutorial covers the workflow of a PyTorch with TorchText project. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model will be simple and achieve poor performance, but this will be improved in the subsequent tutorials.

* 2 - [Upgraded Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

    Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization.

* 3 - [Faster Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/3%20-%20Faster%20Sentiment%20Analysis.ipynb)

    After we've covered all the fancy upgrades to RNNs, we'll look at a different approach that does not use RNNs. More specifically, we'll implement the model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759). This simple model achieves comparable performance as the *Upgraded Sentiment Analysis*, but trains much faster.

* 4 - [Convolutional Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb)

    Finally, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

## Appendices

* A - [Using TorchText with your Own Datasets](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb)

    The tutorials use TorchText's built in datasets. This first appendix notebook covers how to load your own datasets using TorchText.

* B - [A Closer Look at Word Embeddings](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/B%20-%20A%20Closer%20Look%20at%20Word%20Embeddings.ipynb)

    This appendix notebook covers a brief look at exploring the pre-trained word embeddings provided by TorchText.
