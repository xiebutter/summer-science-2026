import torch
import torch.nn as nn
import tiktoken
 
from transformer import TransformerBlock, LayerNorm
 
 
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
 
 
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
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
 
 
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
 
        # predictions
        with torch.no_grad():
            logits = model(idx_cond)
 
        logits = logits[:, -1, :]
 
        # softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
 
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
 
        idx = torch.cat((idx, idx_next), dim=1)
 
    return idx
 
 
def main():
    tokenizer = tiktoken.get_encoding("gpt2")
 
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print("Input batch:\n", batch)
 
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
 
    out = model(batch)
    print("\nOutput shape:", out.shape)
    print(out)
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")
 
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
 
    total_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
 
    model.eval()  # disable dropout
 
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("\nencoded:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
 
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
 
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
 
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("\nDecoded text:", decoded_text)
 
 
if __name__ == "__main__":
    main()