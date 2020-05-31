from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class Sequences(Dataset):
    def __init__(self,path):
        df = pd.read_csv(path)
        # instantiate a vectorizer
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        # fit_transform (vectorize the data)
        self.sequences = self.vectorizer.fit_transform(df.review.tolist())
        # create a list of labels (positive/negative)
        self.labels = df.sentiment.tolist()
        # dictionary with the count of all the words
        self.token2idx = self.vectorizer.vocabulary_
        # list but it's actually a set of all the words
        self.idx2token = {idx: token for token, idx, in self.token2idx.items()}

        # must-define method to match the sparce matrix to the label
    def __getitem__(self, i):
        return self.sequences[i,:].toarray(), self.labels[i]
        # must-define method to get the length of the iterable
    def __len__(self):
        return self.sequences.shape[0]