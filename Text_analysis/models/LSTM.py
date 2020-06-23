import torch.nn as nn
import torch

class LSTMClassifier(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
	bidirectional, dropout):

		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim,
		num_layers=n_layers, bidirectional=bidirectional, dropout=dropout,batch_first=True)
		self.fc = nn.Linear(hidden_dim * 2, output_dim)
		self.act = nn.Sigmoid()

	def forward(self, text, text_lengths):
		embedded = self.embedding(text)
		packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
		packed_output, (hidden, cell) = self.lstm(packed_embedded)
		hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
		dense_outputs=self.fc(hidden)
		outputs=self.act(dense_outputs)
		return outputs