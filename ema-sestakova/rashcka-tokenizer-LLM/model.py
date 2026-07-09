"""
LLM Architecture.
"""
import torch
import torch.nn as nn
from attention_mechanism import MultiHeadAttention

# ----------------------------------------------------------
# Layer Normalization
# ----------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# ----------------------------------------------------------
# GELU Activation
# ----------------------------------------------------------
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# ReLU is a piecewise linear function that outputs the input 
# directly if it is positive; otherwise, it outputs zero.
# ------------------------------
# GELU is a smooth, nonlinear function that approximates ReLU but 
# with a non-zero gradient for almost all negative values

# ----------------------------------------------------------
# Feed Forward Network
# ----------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # input tensor --> linear layer --> GELU activation 
        #       --> linear layer --> output tensor
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

# ----------------------------------------------------------
# Transformer Block 
# ----------------------------------------------------------
# input --> *(shortcut connection) --> layer norm 1 --> masked multi-head attention
#   --> dropout * --> *(shortcut connection) --> layer norm 2
#       --> feed forward (linear layer --> GELU --> linear layer) --> dropout * --> output
# outputs have the same form and dimensions as the inputs
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        # add the original input back
        x = x + shortcut

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        # add the original input back
        x = x + shortcut
        return x

# ----------------------------------------------------------
# GPT Model Architecture
# ----------------------------------------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
  
# ----------------------------------------------------------
# Generating Text
# ----------------------------------------------------------

# INPUT: "Hello, I am", single iteration:
# 1. Encodes text input into four token IDs
# 2. The GPT model returns a matrix consisting of four vectors (rows), 
#       where each vector has 50257 dimensions (columns).
# 3. Extracts the last vector, which corresponds to the next token that the
#       GPT model is supposed to generate
# 4. Converts logits into probability distribution using the softmax function
# 5. Identifies the index position of the largest value, which also 
#       represents the token ID
# 6. Appends token to the previous inputs for the next round

# A text generation function with more diversity. Includes temperature
# sampling and top-k sampling.
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # filters logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        # applies temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # carries out greedy next-token selection as before where temperature
            # scaling is disabled
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            # stops generating early if end-of-sequence token is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
