import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

from binary_accuracy import binary_accuracy
from models.LSTM import LSTMClassifier # own built
import load_data # own built

# DEFINE TRAIN FUNCTION
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad() # delete any gradients
        text, text_lengths = batch.text 
        predictions = model(text, text_lengths).squeeze()
        loss = criterion(predictions, batch.label)
        print(loss.item())
        acc = binary_accuracy(predictions, batch.label) # custom function for binary acc.
        loss.backward() 
        optimizer.step()
        epoch_loss += loss.item()  
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# DEFINE EVAL FUNCTION

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
    
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze()
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# LOAD DATA
path = '/Users/babyhandzzz/Desktop/ELEPH@NT/Datasets/IMDB_toys_datasets'
vectors, emb_dim, vocab_size, train_iter, valid_iter = load_data.load_custom_data(path=path)

# CALLING THE MODEL
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

model = LSTMClassifier(vocab_size, emb_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = False, dropout = dropout)
pretrained_embeddings = vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

N_EPOCHS = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print("epoch: {}".format(epoch))
     
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    
   # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    """
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    """
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')