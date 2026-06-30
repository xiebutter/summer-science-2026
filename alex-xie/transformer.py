from bpe_tokenizer import BPEtokenizer, save_tokens, load_tokens, encode, decode, encode_fast
import torch
from urllib.request import urlopen
import torch.nn as nn
from torch.nn import functional as F


blocksize = 256
batchsize = 64
learning_rate = 3e-4
max_iterations = 5000
eval_interval = 500
eval_iterations = 200
n_embd = 384
heads = 6
n_layer = 6
dropout = 0.2

device = 'cpu'
if torch.mps.is_available():
    device = 'mps'
    print("Running on MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("Running on CUDA (NVIDIA)")
else:
    print("Running on CPU")

def importdata(url):
    with urlopen(url) as response:
        rawdata = response.read().decode('utf-8')
    return rawdata

rawtext = importdata("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")


vocab, merges = load_tokens("shakespeare_token.json")
token_ids = encode_fast(rawtext, vocab, merges)

vocab_size = len(vocab)

data = torch.tensor(token_ids, dtype=torch.long)

n = int(0.9 * len(data))
training_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = training_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i+blocksize] for i in ix])
    y = torch.stack([data[i+1:i+blocksize+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)

        for k in range(eval_iterations):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

    
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blocksize, blocksize)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x 

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(blocksize, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx)
        pos_embd = self.position_embedding_table(torch.arange(T, device = device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -blocksize:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = BigramLanguageModel()
m = model.to(device)
    
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

def train():
    
    for i in range(max_iterations):
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f"\rstep {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        percentage = 100*((i+1)/max_iterations)
        print(f"\r{percentage:.2f}%", end="")
        xb, yb = get_batch('train')

        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\r100.00%\nTraining complete.")
    #context = torch.zeros((1, 1), dtype = torch.long, device=device)

def talk():
    user_input = input("Prompt: ")
    maxtokens = input("Max tokens to generate: ")
    context = torch.tensor(encode_fast(user_input, vocab, merges), dtype=torch.long, device=device).unsqueeze(0)
    text = m.generate(context, max_new_tokens=int(maxtokens))[0].tolist()

    print(decode(text, vocab))