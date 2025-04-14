import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import Counter

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Encoder

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim)     # IF U R USING RNN ---> self.rnn = nn.RNN(emb_dim, hid_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

        '''# RNN version
        outputs, hidden = self.rnn(embedded)
        return hidden, None'''

# Decoder

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim)     # IF U R USING RNN ----> self.rnn = nn.RNN(emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

        ''' # RNN version
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, None'''

# Seq2Seq Model

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        hidden, cell = self.encoder(src)  # RNN version hidden = self.encoder(src)

        input = trg[0, :]  # <sos>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)    # RNN version output, hidden, _ = self.decoder(input, hidden, None)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs

# Data Example

def build_vocab(sentences, min_freq=1):
    counter = Counter(word for sent in sentences for word in sent)
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def sentence_to_tensor(vocab, sentence):
    tokens = [vocab[word] for word in sentence]
    tokens = [vocab['<sos>']] + tokens + [vocab['<eos>']]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)  # [len, 1]


# Load and prepare
data = pd.read_csv("/content/seq2seq.csv", encoding = 'latin-1')
data["Source"] = data["Source"].apply(lambda x: x.lower().strip().split())
data["Target"] = data["Target"].apply(lambda x: x.lower().strip().split())

SRC_vocab = build_vocab(data["Source"])
TRG_vocab = build_vocab(data["Target"])
SRC_itos = {i: s for s, i in SRC_vocab.items()}
TRG_itos = {i: s for s, i in TRG_vocab.items()}

example_src = data["Source"][0]
example_trg = data["Target"][0]
src_tensor = sentence_to_tensor(SRC_vocab, example_src)
trg_tensor = sentence_to_tensor(TRG_vocab, example_trg)

INPUT_DIM = len(SRC_vocab)
OUTPUT_DIM = len(TRG_vocab)
HID_DIM = 256
EMB_DIM = 128
N_EPOCHS = 100
LEARNING_RATE = 0.01

print(data['Source'])
print(data['Target'])
# Train Model

encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(device)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM).to(device)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_vocab['<pad>'])

for epoch in range(N_EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(src_tensor, trg_tensor)  # output: [trg_len, 1, output_dim]
    output_dim = output.shape[-1]

    output = output[1:].view(-1, output_dim)
    trg = trg_tensor[1:].view(-1)

    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Inference
def translate_sentence(model, sentence, src_vocab, trg_vocab, trg_itos, max_len=10):
    model.eval()
    src_tensor = sentence_to_tensor(src_vocab, sentence)
    hidden, cell = model.encoder(src_tensor)

    input = torch.tensor([trg_vocab['<sos>']], device=device)

    translated_tokens = []

    for _ in range(max_len):
        output, hidden, cell = model.decoder(input, hidden, cell)
        top1 = output.argmax(1).item()
        if top1 == trg_vocab['<eos>']:
            break
        translated_tokens.append(trg_itos[top1])
        input = torch.tensor([top1], device=device)

    return translated_tokens

input_sentence = "yes"
tokens = input_sentence.lower().split()  # ['hello', 'how', 'are', 'you']
print("Translation:", translate_sentence(model, tokens, SRC_vocab, TRG_vocab, TRG_itos))

""" Data format 
Source	Target
hello	hola
good morning	buenos d�as
how are you	c�mo est�s
"""
