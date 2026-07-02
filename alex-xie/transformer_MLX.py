from bpe_tokenizer import BPEtokenizer, save_tokens, load_tokens, encode, decode, encode_fast
from urllib.request import urlopen
import mlx.core as mx
import mlx.nn as nn
import pandas as pd
import mlx.optimizers as optim
from datasets import load_dataset

blocksize = 256
batchsize = 32
learn_rate = 3e-4
max_iterations = 5000
eval_interval = 500
eval_iterations = 200
n_embd = 192
heads = 6
n_layer = 6
dropout = 0.3
weightdecay = 1e-4
temp = 0.8


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

# dolly 15k
# df = pd.read_json("hf://datasets/databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl", lines=True)

# tinystories
df = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

categories = ['open_qa', 'general_qa', 'classification', 'closed_qa', 'summarization', 'brainstorming', 'information_extraction', 'summarization', 'creative_writing']

def process_data(df, mode = 'train', max_items = 20000):

    all_x = []
    all_y = []

    for idx, item in enumerate(df):
        if idx >= max_items:
            break
        
        if mode == "finetune":
            # Format the text with your exact special tokens
            system_str = f"<|system|>You are an AI performing a task categorized under: {item['category']}\n"
            user_str = f"<|user|>{item['instruction']}\n"
            context_str = f"<|context|>{item['context']}\n" if item['context'] else ""
            assistant_str = f"<|assistant|>{item['response']}<|end_of_turn|>"
            
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
        elif mode == "train":
            text = item['text']
            full_sequence = encode_fast(text, vocab, merges)

        if len(full_sequence) > blocksize + 1:
            full_sequence = full_sequence[:blocksize + 1]
        
        padding = (blocksize + 1) - len(full_sequence)
        if padding > 0:
            full_sequence += [EOT_ID] * padding
            if mode == "finetune":
                target_sequence += [-100] * padding

        all_x.append(full_sequence)
        if mode == "finetune":
            all_y.append(target_sequence)
        elif mode == "train":
            all_y.append(full_sequence)

    return mx.array(all_x, dtype=mx.int64), mx.array(all_y, dtype=mx.int64)

X_tensor, Y_tensor = process_data(df)

n = int(0.9 * len(X_tensor))
training_X, val_X = X_tensor[:n], X_tensor[n:]
training_Y, val_Y = Y_tensor[:n], Y_tensor[n:]


def get_batch_2D(split):
    data_X = training_X if split == 'train' else val_X
    data_Y = training_Y if split == 'train' else val_Y

    ix = mx.random.randint(0, len(data_X), [batchsize])
    x = data_X[ix, :blocksize]
    y = data_Y[ix, 1:blocksize+1]

    return x, y

def loss_fn(model, x, y):
    logits = model(x)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).mean()

def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = []

        for k in range(eval_iterations):
            X, Y = get_batch_2D(split)
            loss = loss_fn(model, X, Y)
            losses.append(loss.item())

        out[split] = mx.array(losses).mean().item()

    model.train()
    return out

class Head(nn.Module):

    def __init__(self, n_embd, head_size, blocksize, dropout=0.1):
        super().__init__()
        # In MLX, Linear layers expect (input_dims, output_dims)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # No register_buffer needed! Simply assign the array directly.
        self.tril = mx.tril(mx.ones((blocksize, blocksize)))

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        head_size = q.shape[-1]
        wei = (q @ k.transpose(0, 2, 1)) * (head_size**-0.5)        
        mask = self.tril[:T, :T] == 0
        wei = mx.where(mask, float('-inf'), wei)
        
        wei = mx.softmax(wei, axis=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, blocksize, dropout):
        super().__init__()
        self.heads = [Head(n_embd, head_size, blocksize, dropout) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        
        # Apply the projection and residual dropout
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def __call__(self, x): # <-- Changed from forward()
        return self.net(x)
    
class MultiHeadAttention_Parallel(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, blocksize, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Project ALL heads simultaneously using one efficient layer matrix
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        
        self.tril = mx.tril(mx.ones((blocksize, blocksize)))
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B, T, C = x.shape
        
        # 1. Compute projections for all heads at the same time
        k = self.key(x)   # Shape: (B, T, num_heads * head_size)
        q = self.query(x) # Shape: (B, T, num_heads * head_size)
        v = self.value(x) # Shape: (B, T, num_heads * head_size)

        # 2. Reshape & Transpose to separate the heads cleanly on the GPU
        # Target shape for attention: (B, num_heads, T, head_size)
        k = k.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)

        # 3. Batched matrix multiplication across all heads simultaneously
        # k.transpose(0, 1, 3, 2) flips the final two axes of k for the dot product
        wei = (q @ k.transpose(0, 1, 3, 2)) * (self.head_size ** -0.5)        
        
        # Apply causal mask (automatically broadcasts across batch and head dimensions)
        mask = self.tril[:T, :T] == 0
        wei = mx.where(mask, float('-inf'), wei)
        wei = mx.softmax(wei, axis=-1)
        wei = self.dropout(wei)

        # 4. Multiply by values: (B, H, T, T) @ (B, H, T, head_size) -> (B, H, T, head_size)
        out = wei @ v
        
        # 5. Transpose back and flatten the heads back into the embedding dimension
        # (B, H, T, head_size) -> (B, T, H, head_size) -> (B, T, H * head_size)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)

        # Apply output projection and residual dropout
        return self.dropout(self.proj(out))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, blocksize, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        
        # Pass the configurations directly to your submodules!
        self.sa_heads = MultiHeadAttention_Parallel(n_head, head_size, n_embd, blocksize, dropout)
        self.ffwd = FeedForward(n_embd, dropout)

        # In MLX, use 'dims=n_embd' for LayerNorm layers
        self.ln1 = nn.LayerNorm(dims=n_embd)
        self.ln2 = nn.LayerNorm(dims=n_embd)

    def __call__(self, x): # <-- Changed from forward()
        # Pre-LN residual connections work exactly the same way!
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

class XieLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, blocksize, n_layer, n_head, dropout=0.1):
        super().__init__()
        self.blocksize = blocksize
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(blocksize, n_embd)
        
        # MLX automatically tracks sub-modules listed inside a standard Python list!
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, blocksize, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(dims=n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):  # <-- Changed from forward() and stripped targets parameter
        B, T = idx.shape

        token_embd = self.token_embedding_table(idx)
        
        # Use mx.arange to construct the position matrix dynamically
        pos_embd = self.position_embedding_table(mx.arange(T))
        
        x = token_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Ensure the model is toggled to eval mode for text generation
        self.eval()

        for _ in range(max_new_tokens):
            # Crop the sequence length back to the maximum supported block size
            idx_cond = idx[:, -self.blocksize:]

            # Forward pass: just compute the raw logits
            logits = self(idx_cond)
            
            # Focus completely on the last token position's logit distribution
            logits = logits[:, -1, :] / temperature
            
            # Native categorical sampling in MLX (does not require an explicit softmax first!)
            idx_next = mx.random.categorical(logits, num_samples=1)
            
            # Concatenate along the sequence axis (axis 1)
            idx = mx.concatenate([idx, idx_next], axis=1)

        self.train()
        return idx

model = XieLanguageModel(vocab_size=vocab_size, n_embd=n_embd, blocksize=blocksize, n_layer=n_layer, n_head=heads, dropout=dropout)
    
optimizer = optim.AdamW(learning_rate=learn_rate, weight_decay=weightdecay)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

def train():
    
    print("Starting MLX training loop...")
    
    for i in range(max_iterations):
        if i % eval_interval == 0:
            # Computes evaluation metrics using your converted MLX estimate_loss function
            losses = estimate_loss()
            print(f"\rstep {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        percentage = 100 * ((i + 1) / max_iterations)
        print(f"\r{percentage:.2f}%", end="")
        
        # Ensure your batch function returns native MLX arrays, not PyTorch tensors!
        xb, yb = get_batch_2D('train')

        # Simultaneously compute the scalar loss and the gradient dictionary
        loss, grads = loss_and_grad_fn(model, xb, yb)
        
        # Update the decoupled optimizer and model weights
        optimizer.update(model, grads)
        
        # CRITICAL: Force MLX's lazy execution graph to compute on the Apple Silicon GPU
        mx.eval(model.state, optimizer.state)

    print(f"\r100.00%\nTraining complete.")

    # Save weights using high-performance native Safetensors format
    model.save_weights("model_weights.safetensors")
    print("Model weights saved to 'model_weights.safetensors'")

def talk():
    # Loop continuously to allow real-time chat sessions without restarting
    while True:
        print(f"\nCategories: {categories}")
        category = input("Category: ")
        
        if category.lower() not in categories:
            if category.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            print("Category defaulted to 'general_qa'")
            category = 'general_qa'
            
        system_prompt = f"<|system|>You are an AI performing a task categorized under: {category}\n"

        user_input = input("Prompt: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        maxtokens = input("Max tokens to generate: ")
        if not maxtokens.isdigit():
            print("Max tokens defaulted to 100")
            maxtokens = 100
        else:
            maxtokens = int(maxtokens)

        # Build the exact prompt formatting matching our training structure
        full_prompt = f"{system_prompt}<|user|>{user_input}\n<|assistant|>"

        # 1. Tokenize prompt to a flat list using your clean fast encoder
        context_tokens = encode_fast(full_prompt, vocab, merges)
        
        # 2. Convert to an MLX array and add a batch axis (equivalent to unsqueeze)
        context = mx.array([context_tokens]) # Shape: (1, sequence_length)
        
        # Crop context back to your model's maximum supported block size
        if context.shape[1] > blocksize:
            context = context[:, -blocksize:]

        # 3. Generate text using your MLX model's random sampling routine
        # MLX model generation passes return the complete sequence directly!
        generated_tensor = model.generate(context, max_new_tokens=maxtokens, temperature=temp)
        
        # Force evaluation of the lazy graph layout on the Apple Silicon GPU
        mx.eval(generated_tensor)

        # 4. Extract generated tokens safely by converting to a standard Python list
        generated_text = generated_tensor[0].tolist()

        # Slice the array by your prompt's original token length to isolate the model's response
        prompt_length = len(context_tokens)
        response_tokens = generated_text[prompt_length:]

        # 5. Decode back to strings and isolate clean text from padding/control elements
        raw_response = decode(response_tokens, vocab)
        clean = raw_response.split("<|end_of_turn|>")[0].strip()
        
        print(f"\nResponse: {clean}")