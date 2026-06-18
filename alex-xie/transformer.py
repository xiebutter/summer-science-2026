from bpe_tokenizer import BPEtokenizer, save_tokens, load_tokens, encode, decode
import torch
from urllib.request import urlopen

def importdata(url):
    with urlopen(url) as response:
        rawdata = response.read().decode('utf-8')
    return rawdata

rawtext = importdata("https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt")

vocab, merges = load_tokens("hamlet_token.json")
token_ids = encode(rawtext, vocab, merges)

data = torch.tensor(token_ids, dtype=torch.long)

n = int(0.9 * len(data))
training_data = data[:n]
val_data = data[n:]

blocksize = 8
batchsize = 4

def get_batch(split):
    data = training_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i+blocksize] for i in ix])
    y = torch.stack([data[i+1:i+blocksize+1] for i in ix])

    return x, y

xb, yb = get_batch('train')
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
m = BigramLanguageModel(vocab_size=len(vocab))
logits, loss = m(xb, yb)

print(logits.shape)
print(loss)