import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from transformer_block import Block

text_input = [
    "hi! how are you doing today?",
    "what is the meaning of life?",
    "I am a potato",
    "He likes coffee",
    "I enjoy hiking",
    "You are handsome",
    "I definitely need a longer sentence",
    "Your friend is a musician",
    "Something to say here",
    "I learn language"
]

text_input = [s + " <END>" for s in text_input]
text = " ".join(text_input)

words = list(set(text.split()))
vocab_size = len(words)
print(vocab_size)

word2idx = {w: i for i, w in enumerate(words)}

data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

# parameters
block_size = 6 # the maximum number of tokens the model can attend to at once
embed_dim = 32 # 32 values for each word
n_heads = 2
n_layers = 2
lr = 0.0001
epochs = 1500

def get_batch(batch_size=16):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim) # (37, 32)
        self.position_embed = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_heads) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos_emb = self.position_embed(torch,arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.block(x)
        x = self.ln_f(x) # final layer norm, stabilizes the values before projecting to logits
        logits = self.head(x) # projects each token's hidden state into a score over the entire vocab
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T)) # how wrong your predicted distribution is compared to the true answer
        return logits, loss


