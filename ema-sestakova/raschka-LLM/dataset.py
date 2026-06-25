"""
Data Preparation and Sampling.
"""
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

# ----------------------------------------------------------
# Dataloader
# ----------------------------------------------------------
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, 
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
        #  specified batch_size to prevent loss spikes during training.
        drop_last=drop_last,
        # the number of CPU processes to use for preprocessing
        num_workers=num_workers
    )
    return dataloader

# ----------------------------------------------------------

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 90% of data used for training, 10% for validation
def split_data(text_data, train_ratio=0.9):
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    return train_data, val_data

def prepare_dataloaders(file_path, batch_size, max_length,
                        train_ratio=0.9, stride=None, num_workers=0,):
    if stride is None:
        stride = max_length
    text_data = load_text(file_path)
    train_data, val_data = split_data(text_data, train_ratio=train_ratio)
    train_loader = create_dataloader(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader

# prints token information from the dataset
def print_dataset(file_path, tokenizer, train_ratio=0.9):
    text_data = load_text(file_path)
    train_data, val_data = split_data(text_data, train_ratio=train_ratio)
    train_tokens = tokenizer.encode(train_data)
    val_tokens = tokenizer.encode(val_data)
    unique_chars = sorted(set(text_data))
    return {
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "unique_chars": unique_chars,
    }
