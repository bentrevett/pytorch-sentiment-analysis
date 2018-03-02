from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

from tqdm import tqdm

import random
import os

import utils
import models

#constants
N_EPOCHS = 25 #can set this high as we have early stopping patience
#INIT_LR = 1
#MAX_LENGTH = 100
#SCHEDULER_PATIENCE = 0
#EMB_DIM = 64
#HID_DIM = 128
#RNN_TYPE = 'LSTM'
#N_LAYERS = 2
#BIDIRECTIONAL = True
#DROPOUT = 0.5
USE_CUDA = torch.cuda.is_available()
#WEIGHT_AVERAGE = True
#SEED = 1234

while True:

    #pick parameters
    OPTIMIZER = random.choice(['adam', 'rmsprop', 'sgd'])
    OPTIMIZER_EXTRA_PARAM_1 = random.choice([0.8, 0.85, 0.9, 0.99, 0.999]) #beta1 for adam, alpha for rmsprop
    OPTIMIZER_EXTRA_PARAM_2 = random.choice([0.8, 0.85, 0.9, 0.99, 0.999]) #beta2 for adam
    BATCH_SIZE = random.choice([32, 64, 128, 256])
    INIT_LR = random.choice([1, 0.5, 0.1, 0.005, 0.001, 0.0005, 0.0001])
    MOMENTUM = random.choice([0, 0.9, 0.999])
    MOMENTUM_EXTRA_PARAM = random.choice([True, False])
    MAX_LENGTH = random.choice([None, 100, 200])
    SCHEDULER_PATIENCE = random.choice([None, 0, 1, 2, 3])
    EMB_DIM = random.choice([50, 100, 200, 300])
    HID_DIM = random.choice([50, 100, 200, 300])
    RNN_TYPE = random.choice(['LSTM', 'GRU'])
    N_LAYERS = random.choice([1, 2])
    BIDIRECTIONAL = random.choice([True, False])
    DROPOUT = random.choice([0.1, 0.25, 0.5, 0.75])
    WEIGHT_AVERAGE = random.choice([True, False])
    FREEZE_EMBEDDING = random.choice([True, False])

    prev_save_name = ''
    param_name = f'OP:{OPTIMIZER},O1:{OPTIMIZER_EXTRA_PARAM_1},O2:{OPTIMIZER_EXTRA_PARAM_2},BS:{BATCH_SIZE},IR:{INIT_LR},MO:{MOMENTUM},ME:{MOMENTUM_EXTRA_PARAM},ML:{MAX_LENGTH},SP:{SCHEDULER_PATIENCE},ED:{EMB_DIM},HD:{HID_DIM},RT:{RNN_TYPE},NL:{N_LAYERS},BI:{BIDIRECTIONAL},DO:{DROPOUT},WA:{WEIGHT_AVERAGE},FE:{FREEZE_EMBEDDING}'

    print(f'PARAMS:{param_name}')

    # set up fields
    TEXT = data.Field(batch_first=True, fix_length=MAX_LENGTH)
    LABEL = data.Field(sequential=False, unk_token=None)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # print information about the data
    print('train.fields', train.fields)
    print('len(train)', len(train))
    print('vars(train[0])', vars(train[0]))

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=EMB_DIM))
    LABEL.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    print(LABEL.vocab.stoi)

    # make iterator for splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), 
        batch_size=32, 
        sort_key=lambda x: len(x.text), 
        sort_within_batch=True, 
        repeat=False)

    train_dl = utils.BatchWrapper(train_iter, 'text', ['label'])
    test_dl = utils.BatchWrapper(test_iter, 'text', ['label'])

    model = models.RNNClassification(len(TEXT.vocab),
                                    1,
                                    EMB_DIM,
                                    HID_DIM,
                                    RNN_TYPE,
                                    N_LAYERS,
                                    BIDIRECTIONAL,
                                    DROPOUT)

    if OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=MOMENTUM, nesterov=False if MOMENTUM == 0 else MOMENTUM_EXTRA_PARAM)
    elif OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=INIT_LR, betas=(OPTIMIZER_EXTRA_PARAM_1, OPTIMIZER_EXTRA_PARAM_2))
    elif OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=INIT_LR, alpha=OPTIMIZER_EXTRA_PARAM_1, centered=MOMENTUM_EXTRA_PARAM)
    else:
        raise ValueError(f'Optimizer not found! {OPTIMIZER}')
    
    if SCHEDULER_PATIENCE is not None:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, verbose=True)
    
    criterion = nn.BCEWithLogitsLoss()

    if USE_CUDA:
        model = model.cuda()
        critierion = criterion.cuda()

    if FREEZE_EMBEDDING:
        model.embedding.weight.requires_grad = False

    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, N_EPOCHS+1):

        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for x, y in tqdm(train_dl, desc='Train'):

            optimizer.zero_grad()

            preds = model(x)
            loss = criterion(preds, y)

            if WEIGHT_AVERAGE:
                pre_backprop_params = nn.utils.parameters_to_vector(model.parameters())

            loss.backward()

            optimizer.step()

            if WEIGHT_AVERAGE:
                post_backprop_params = nn.utils.parameters_to_vector(model.parameters())
                averaged_params = (pre_backprop_params+post_backprop_params)/2
                nn.utils.vector_to_parameters(averaged_params, model.parameters()) 

            acc = utils.binary_accuracy(preds, y)

            epoch_loss += loss.data[0]
            epoch_acc += acc.data[0]

        train_acc = epoch_acc / len(train_dl)
        train_loss = epoch_loss / len(train_dl)

        epoch_loss = 0
        epoch_acc = 0 
        
        model.eval()

        for x, y in tqdm(test_dl, desc='Valid'):

            preds = model(x)
            loss = criterion(preds, y)

            acc = utils.binary_accuracy(preds, y)

            epoch_loss += loss.data[0] 
            epoch_acc += acc.data[0]

        test_acc = epoch_acc / len(test_dl)
        test_loss = epoch_loss / len(test_dl)

        if SCHEDULER_PATIENCE is not None:
            scheduler.step(test_loss)

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            if prev_save_name != '':
                os.remove(prev_save_name)
            save_name = f'saves/L:{test_loss:.3f},A:{test_acc*100:.2f}%,E:{epoch},{param_name}.pt'
            torch.save(model.state_dict(), save_name)
            prev_save_name = save_name

        if test_acc<0.4:
            print('Accuracy Crashed. Breaking out.')
            break