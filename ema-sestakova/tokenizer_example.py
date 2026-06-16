# Helper functions
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):        
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0        
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])                
            i += 1
    return newids

# -----------------------------------------------------------------------------------------------------------------------
# BasicTokenizer Class

class BasicTokenizer():
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
        self._build_vocab()

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids) # concatenate bytes together
        text = tokens.decode("utf-8", errors="replace") # not all tokens are standard utf-8, so use errors="replace"
        return text
    
    def _build_vocab(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

# -----------------------------------------------------------------------------------------------------------------------
# Testing

# text = "ABDCABECAB"
# original_tokens = list(text.encode("utf-8"))
# vocab_size_prev = 256

# tokenizer = BasicTokenizer()
# vocab_size_target = 260
# tokenizer.train(text, vocab_size=vocab_size_target)

# vocab_size_after = len(tokenizer.vocab)
# encoded = tokenizer.encode(text)
# decoded = tokenizer.decode(encoded)

# print("Before encoding:")
# print(f"text: {text}")
# print(f"vocab size: {vocab_size_prev}")
# print(f"number of tokens: {len(original_tokens)}")

# print("\nAfter encoding:")
# print(f"decoded text: {decoded}")
# print(f"vocab size: {vocab_size_after}")
# print(f"number of tokens: {len(encoded)}")

text = "ABDCABECAB"
def show_tokens(ids, tokenizer):
    return " ".join(tokenizer.vocab[i].decode("utf-8", errors="replace") for i in ids)

original_tokens = list(text.encode("utf-8"))
vocab_size_prev = 256
tokenizer = BasicTokenizer()
vocab_size_target = 258
tokenizer.train(text, vocab_size=vocab_size_target)
vocab_size_after = len(tokenizer.vocab)
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
original_display = " ".join(text)
encoded_display = show_tokens(encoded, tokenizer)

print("Before encoding:")
print(f"text: {original_display}")
print(f"vocab size: {vocab_size_prev}")
print(f"number of tokens: {len(original_tokens)}")

print("\nAfter encoding:")
print(f"text: {encoded_display}")
print(f"vocab size: {vocab_size_after}")
print(f"number of tokens: {len(encoded)}")
