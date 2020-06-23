import torch 
from torchtext import data
import random

path_ = '/Users/babyhandzzz/Desktop/ELEPH@NT/Datasets/clean_IMDB.csv'
SEED = 2020
torch.manual_seed(SEED)
BATCH_SIZE = 4

def load_custom_data(path):
    
    """
    This function returns:
     . embedding vectors 
     . embedding vectors' length
     . vocab length
     . train iterator
     . test iterator    
    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(tokenize=tokenize, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    fields = [('text',TEXT),('label',LABEL)]
    training_data = data.TabularDataset(path = path_,format = 'csv',fields = fields,skip_header = True)
    train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))

    TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")  
    LABEL.build_vocab(train_data)
    
    EMB_VECTOR_LENGTH = TEXT.vocab.vectors.shape[1]
    
    train_iterator, valid_iterator = data.BucketIterator.splits((train_data,valid_data),
    batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), sort_within_batch=True)
    print('Data loading is complete')
    return TEXT.vocab.vectors, EMB_VECTOR_LENGTH,len(TEXT.vocab), train_iterator, valid_iterator
