from bpe_tokenizer import BPEtokenizer, save_tokens, load_tokens, encode, decode, encode_fast
import torch
from urllib.request import urlopen
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

blocksize = 128
batchsize = 32
learning_rate = 3e-4
max_iterations = 5000
eval_interval = 500
eval_iterations = 200
n_embd = 128
heads = 4
n_layer = 4
dropout = 0.3
weightdecay = 1e-4
temp = 0.8

device = 'cpu'
if torch.mps.is_available():
    device = 'mps'
    print("Running on MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = 'cuda'
    print("Running on CUDA (NVIDIA)")
else:
    print("Running on CPU")

# def importdata(url):
#     with urlopen(url) as response:
#         rawdata = response.read().decode('utf-8')
#     return rawdata

# rawtext = importdata("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")


# vocab, merges = load_tokens("shakespeare_token.json")


vocab, merges = load_tokens("dolly_tokens.json")

vocab_size = len(vocab)

SYSTEM_ID = vocab['<|system|>']
USER_ID = vocab['<|user|>']
CONTEXT_ID = vocab['<|context|>']
ASSISTANT_ID = vocab['<|assistant|>']
EOT_ID = vocab['<|end_of_turn|>']

df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

categories = ['open_qa', 'general_qa', 'classification', 'closed_qa', 'summarization', 'brainstorming', 'information_extraction', 'summarization', 'creative_writing']

def process_data(df):

    all_x = []
    all_y = []

    for _, row in df.iterrows():
        # Format the text with your exact special tokens
        system_str = f"<|system|>You are an AI performing a task categorized under: {row['category']}\n"
        user_str = f"<|user|>{row['instruction']}\n"
        context_str = f"<|context|>{row['context']}\n" if row['context'] else ""
        assistant_str = f"<|assistant|>{row['response']}<|end_of_turn|>"
        
        # Encode each piece separately so we know exactly where the assistant's response begins
        ids_system = encode_fast(system_str, vocab, merges)
        ids_user = encode_fast(user_str, vocab, merges)
        ids_context = encode_fast(context_str, vocab, merges)
        ids_assistant = encode_fast(assistant_str, vocab, merges)

        full_sequence = ids_system + ids_user + ids_context + ids_assistant

        if len(full_sequence) > blocksize + 1:
            full_sequence = full_sequence[:blocksize + 1]
        
        target_sequence = [-100] * len(full_sequence)
        assistant_start_idx = len(ids_system) + len(ids_user) + len(ids_context)

        for i in range(assistant_start_idx, len(full_sequence)):
            target_sequence[i] = full_sequence[i]
        
        padding = (blocksize + 1) - len(full_sequence)
        if padding > 0:
            full_sequence += [EOT_ID] * padding
            target_sequence += [-100] * padding

        all_x.append(full_sequence)
        all_y.append(target_sequence)

    return torch.tensor(all_x, dtype=torch.long), torch.tensor(all_y, dtype=torch.long)

X_tensor, Y_tensor = process_data(df)

n = int(0.9 * len(X_tensor))
training_X, val_X = X_tensor[:n], X_tensor[n:]
training_Y, val_Y = Y_tensor[:n], Y_tensor[n:]

# def get_batch_1D(split):
#     data = training_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - blocksize, (batchsize,))
#     x = torch.stack([data[i:i+blocksize] for i in ix])
#     y = torch.stack([data[i+1:i+blocksize+1] for i in ix])

#     x, y = x.to(device), y.to(device)

#     return x, y

def get_batch_2D(split):
    data_X = training_X if split == 'train' else val_X
    data_Y = training_Y if split == 'train' else val_Y

    ix = torch.randint(len(data_X), (batchsize,))
    x = torch.stack([data_X[i, :blocksize] for i in ix])
    y = torch.stack([data_Y[i, 1:blocksize+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)

        for k in range(eval_iterations):
            X, Y = get_batch_2D(split)
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

        head_size = q.shape[-1]
        wei = q @ k.transpose(-2, -1) * (head_size**-0.5)
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

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

class XieLanguageModel(nn.Module):
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
    
    def generate(self, idx, max_new_tokens, temperature):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -blocksize:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

model = XieLanguageModel()
m = model.to(device)
    
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weightdecay)

def train():
    
    for i in range(max_iterations):
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(f"\rstep {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        percentage = 100*((i+1)/max_iterations)
        print(f"\r{percentage:.2f}%", end="")
        xb, yb = get_batch_2D('train')

        logits, loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"\r100.00%\nTraining complete.")

    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved to 'model_weights.pth'")

def talk():
    while True:
        
        print(f"Categories: {categories}")
        category = input("Category: ")
        if category.lower() not in categories:
            if category.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            print("Category default to 'general_qa'")
            category = 'general_qa'
        system_prompt = f"<|system|>You are an AI performing a task categorized under: {category}\n"

        user_input = input("Prompt: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        maxtokens = input("Max tokens to generate: ")

        if not maxtokens.isdigit():
            print("Max tokens default to 100")
            maxtokens = 100
        else:
            maxtokens = int(maxtokens)

        full_prompt = f"{system_prompt}<|user|>{user_input}\n<|assistant|>"

        context_tokens = encode_fast(full_prompt, vocab, merges)
        context = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
        if context.shape[1] > blocksize:
            context = context[:, -blocksize:]

        generated_text = m.generate(context, max_new_tokens=int(maxtokens), temperature=temp)[0].tolist()

        prompt_length = len(context_tokens)
        response_tokens = generated_text[prompt_length:]

        raw_response = decode(response_tokens, vocab)

        clean = raw_response.split("<|end_of_turn|>")[0].strip()
        print(f"Response: {clean}")