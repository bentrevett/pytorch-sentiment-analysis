# PyTorch Sentiment Analysis

This repo contains tutorials covering how to do sentiment analysis using [PyTorch](https://github.com/pytorch/pytorch) and [TorchText](https://github.com/pytorch/text).

The first 2 tutorials will cover getting started with the de facto approach to sentiment analysis: recurrent neural networks (RNNs). Subsequent tutorials will cover different approaches.

## Tutorials

* 1 - [Simple Sentiment Analysis]()

    This tutorial covers the workflow of a PyTorch with TorchText project. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model will be simple and achieve poor performance, but this will be improved in the subsequent tutorials.

* 2 - [Upgraded Sentiment Analysis]()

    Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization.

* 3 - [Faster Sentiment Analysis]()

    After we've covered all the fancy upgrades to RNNs, we'll look at a different approach that does not use RNNs. More specifically, we'll implement the model from [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759). This simple model achieves comparable performance as the *Upgraded Sentiment Analysis*, but trains much faster.

* 4 - [Convolutional Sentiment Analysis]() (WIP)

    Finally, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882). 

## Appendices

* A - [Using TorchText with your Own Datasets](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb)

    The tutorials use TorchText's built in datasets. This first appendix notebook covers how to load your own datasets using TorchText.

* B - [A Closer Look at Word Embeddings]() (WIP)
    
    This appendix notebook covers a brief look at exploring the pre-trained word embeddings provided by TorchText.
