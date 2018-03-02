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

import utils
import models

#constants
N_EPOCHS = 50 #can set this high as we have early stopping patience
N_BORN_AGAIN = 5
INIT_LR = 1
SCHEDULER_PATIENCE = 1
PATIENCE = 2
EMB_DIM = 300
HID_DIM = 300
RNN_TYPE = 'LSTM'
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.5
USE_CUDA = torch.cuda.is_available()
WEIGHT_AVERAGE = True
SEED = 1234

utils.set_seeds(SEED)

# set up fields
TEXT = data.Field(batch_first=True, fix_length=100)
LABEL = data.Field(sequential=False, unk_token=None)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=EMB_DIM))
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

optimizer = optim.RMSprop(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, verbose=True)
criterion = nn.BCEWithLogitsLoss()
ba_criterion = nn.MSELoss()

if USE_CUDA:
    model = model.cuda()
    critierion = criterion.cuda()
    ba_criterion = ba_criterion.cuda()

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

    scheduler.step(test_loss)

    print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f'Patience count {patience_counter}/{PATIENCE}')

    if patience_counter > PATIENCE:
        print('Patience exceeded!')
        break

print(f'First network trained. Beginning Born Again Cycle.')

for i in range(1, N_BORN_AGAIN+1):

    print(f'Born Again Cycle: {i}')

    best_test_loss = float('inf')
    patience_counter = 0

    old_model = model
    
    old_model.eval() #turn off dropout for old model, does this need to be true?

    #freeze parameters of the old model
    for param in old_model.parameters():
        param.requires_grad = False

    model = models.RNNClassification(len(TEXT.vocab),
                                    1,
                                    EMB_DIM,
                                    HID_DIM,
                                    RNN_TYPE,
                                    N_LAYERS,
                                    BIDIRECTIONAL,
                                    DROPOUT)

    #need to tell optimizer to only update the new model
    #if you don't do this, the optimizer will update the old model
    optimizer = optim.Adam(model.parameters())
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, verbose=True)

    if USE_CUDA:
        model = model.cuda()

    for epoch in range(1, N_EPOCHS+1):

        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for x, y in tqdm(train_dl, desc='Train'):

            optimizer.zero_grad()

            preds = model(x)

            old_preds = old_model(x)
            old_preds = Variable(old_preds.data, requires_grad = False) #need to do this to use outputs as targets
                        
            loss = criterion(preds, y) + ba_criterion(preds, old_preds)
            
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

            old_preds = old_model(x)
            old_preds = Variable(old_preds.data, requires_grad = False)
            
            loss = criterion(preds, y) + ba_criterion(preds, old_preds)

            acc = utils.binary_accuracy(preds, y)

            epoch_loss += loss.data[0] 
            epoch_acc += acc.data[0]

        test_acc = epoch_acc / len(test_dl)
        test_loss = epoch_loss / len(test_dl)

        scheduler.step(test_loss)

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Patience count {patience_counter}/{PATIENCE}')

        if patience_counter > PATIENCE:
            print('Patience exceeded!')
            break
