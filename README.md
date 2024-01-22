# PyTorch Sentiment Analysis

This repo contains tutorials covering understanding and implementing sequence classification models using [PyTorch](https://github.com/pytorch/pytorch), with Python 3.9. Specifically, we'll train models to predict sentiment from movie reviews.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-sentiment-analysis/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

Install the required dependencies with: `pip install -r requirements.txt --upgrade`.

## Tutorials

-   1 - [Neural Bag of Words](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/1%20-%20Neural%20Bag%20of%20Words.ipynb)

    This tutorial covers the workflow of a sequence classification project with PyTorch. We'll cover the basics of sequence classification using a simple, but effective, neural bag-of-words model, and how to use the datasets/torchtext libaries to simplify data loading/preprocessing.

-   2 - [Recurrent Neural Networks](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/2%20-%20Recurrent%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/2%20-%20Recurrent%20Neural%20Networks.ipynb)

    Now we have the basic sequence classification workflow covered, this tutorial will focus on improving our results by switching to a recurrent neural network (RNN) model. We'll cover the theory behind RNNs, and look at an implementation of the long short-term memory (LSTM) RNN, one of the most common variants of RNN.

-   3 - [Convolutional Neural Networks](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/3%20-%20Convolutional%20Neural%20Networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/3%20-%20Convolutional%20Neural%20Networks.ipynb)

    Next, we'll cover convolutional neural networks (CNNs) for sentiment analysis. This model will be an implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

-   4 - [Transformers](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/main/4%20-%20Transformers.ipynb)

    Finally, we'll show how to use the transformers library to load a pre-trained transformer model, specifically the BERT model from [this](https://arxiv.org/abs/1810.04805) paper, and use it for sequence classification.

## Legacy Tutorials

Previous versions of these tutorials used features from the torchtext library which are no longer available. These are stored in the [legacy](https://github.com/bentrevett/pytorch-sentiment-analysis/tree/main/legacy) directory.

## References

Here are some things I looked at while making these tutorials. Some of it may be out of date.

-   http://anie.me/On-Torchtext/
-   http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
-   https://github.com/spro/practical-pytorch
-   https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
-   https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
-   https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py
-   https://github.com/Shawn1993/cnn-text-classification-pytorch
