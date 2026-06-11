import os
import random
import math
import json

import unicodedata
import re
from collections import Counter, defaultdict
from heapq import heappush, heappop

test_text = "Hello, how are you? Hello, I am doing good."

#text normalization: "Hëllô, hòw âré yôu?" --> [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), 
#                                               ('Ġyou', (14, 18)), ('?', (18, 19)）]
#spaces replaced by Ġ to make it easier to read later on

def normalize(text):
    text = text.strip() 
    norm = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Replace spaces with Ġ upfront so the regex can see them clearly
    norm = norm.replace(" ", "Ġ")
    
    pattern = re.compile(r'Ġ?[a-zA-Z0-9]+|Ġ?[^Ġa-zA-Z0-9]|Ġ+')
    
    result = []
    for match in pattern.finditer(norm):
        token = match.group()
        if token.strip("Ġ") == "":
            continue
        start, end = match.span()
        result.append((token, (start, end)))
        
    return result

#convert normalized text into a list of characters
def convertChar(normText):
    chars = []
    for tok, span in normText:
        word = []
        for char in tok:
            word.append(char)
        chars.append(word)
    return chars

#count the frequency of each character pair
def pairCounter(chars):
    counts = Counter()
    for word in chars:
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counts[pair] += 1
    return counts

#merges common character pairs
def mergePairs(charList):
    merges = {}
    counts = pairCounter(charList)

    if not counts:
        return charList, merges
    
    maxCount = max(counts.values())
    bestPairs = [pair for pair, count in counts.items() if count == maxCount]

    newList = []
    for pair in bestPairs:
        first, second = pair
        currentList = []

        for word in charList:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            currentList.append(new_word)

        merges[pair] = "".join(pair)
        charList = currentList
    
    return charList, merges

#BPE training
def BPEtokenizer(rawtext, targetsize):
    pretokens = convertChar(normalize(rawtext))

    base_vocab = set()
    for word in pretokens:
        for char in word:
            base_vocab.add(char)
    
    vocab = {char: idx for idx, char in enumerate(sorted(base_vocab))}
    merges = {}

    print(f"Starting Byte Pair Encoding training. Base vocab size: {len(vocab)}")

    while len(vocab) < targetsize:
        pretokens, newmerges = mergePairs(pretokens)

        if not newmerges:
            print("No more merges. Training stopped")
            break

        for pair, merged in newmerges.items():
            if len(vocab) >= targetsize:
                print("Target vocab size reached. Training stopped")
                break
            merges[pair] = merged
            vocab[merged] = len(vocab)
        print(f"Current vocab size: {len(vocab)} / {targetsize}")
    return vocab, merges

#save token data
def save_tokens(vocab, merges, filename="token_data.json"):
    token_data = {
        "vocab": vocab,
        "merges": {str(k): v for k, v in merges.items()}
    }
    with open(filename, "w") as f:
        json.dump(token_data, f, indent = 4)
    print(f"Token data saved to {filename}")

#load token data
def load_tokens(filename="token_data.json"):
    with open(filename, "r") as f:
        token_data = json.load(f)

    token_data["merges"] = {eval(k): v for k, v in token_data["merges"].items()}

    print(f"Token data loaded from {filename}")
    return token_data["vocab"], token_data["merges"]

#encode text with token ids
def encode(rawtext, merges, vocab):
    charList = convertChar(normalize(rawtext))
    
    # apply your learned merges in the exact order they were trained
    for pair, merged_string in merges.items():
        first, second = pair
        currentList = []
        
        for word in charList:
            new_word = []
            i = 0
            while i < len(word):
                # Look for the target pair and glue them together
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            currentList.append(new_word)

        charList = currentList
    token_ids = []
    for word in charList:
        for token in word:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                print(f"Warning: Character '{token}' not found in vocabulary. Skipping.")
                
    return token_ids

#decode text from token ids
def decode(token_ids, vocab):
    id_to_token = {idx: token for token, idx in vocab.items()}
    
    # Convert the IDs back into their string fragments
    fragments = [id_to_token[uid] for uid in token_ids]

    merged_text = "".join(fragments)
    clean_text = merged_text.replace("Ġ", " ")
    
    return clean_text

#tokenize hamlet

# from urllib.request import urlopen
# url = "https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt"
# with urlopen(url) as response:
#     rawdata = response.read().decode('utf-8')