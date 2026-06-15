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