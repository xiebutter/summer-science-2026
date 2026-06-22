# Data preparation and sampling

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # tokenizes the entire text
        token_ids = tokenizer.encode(txt)
        # uses a sliding window to chunk the book into overlapping 
        # sequences of max length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    # returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    # initializes the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # creates dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # drop_last=True drops the last batch if it is shorter than the
        # specified batch_size to prevent loss spikes during training.
        drop_last=drop_last,
        # The number of CPU processes to use for preprocessing
        num_workers=num_workers
    )
    return dataloader

# ---------------------------------------------------------------------------

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Dataloader
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length,
                                  stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# First embedding layer
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# Second embedding layer
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
