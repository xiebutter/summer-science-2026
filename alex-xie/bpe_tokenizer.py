import os
import random
import math
import json

import unicodedata
import re
from collections import Counter, defaultdict

test_text = "Hello, how are you? Hello, I am doing good."

# simplify using basic python (intro to cs class-level)

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
def BPEtokenizer(rawtext, targetsize = 5000):
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
def encode(rawtext, vocab, merges):
    charList = convertChar(normalize(rawtext))
    
    # apply your learned merges in the exact order they were trained
    print("Encoding text...")
    perc = 0
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
        perc += 1
        total = perc/len(merges.items())
        print(f"\r{100*total:.1f}%", end="")

    token_ids = []
    for word in charList:
        for token in word:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                print(f"Warning: Character '{token}' not found in vocabulary. Skipping.")
    
    print("\nRaw text encoded", end="\n")          
    return token_ids

#decode text from token ids
def decode(token_ids, vocab):
    id_to_token = {idx: token for token, idx in vocab.items()}
    
    # Convert the IDs back into their string fragments
    fragments = [id_to_token[uid] for uid in token_ids]

    merged_text = "".join(fragments)
    clean_text = merged_text.replace("Ġ", " ")
    
    return clean_text

def encode_fast(rawtext, vocab, merges):
    # 1. Normalize and clean input
    # Assuming convertChar returns a list of words, where each word is a list/tuple of chars
    words = convertChar(normalize(rawtext))
    
    # Create a fast lookup rank dictionary for merges: { (first, second): rank_id }
    # Lower rank means it was merged earlier in training (higher priority)
    merge_ranks = {pair: i for i, pair in enumerate(merges.keys())}
    
    # Cache to store already-tokenized unique words
    word_cache = {}
    
    def encode_word(word_tuple):
        """Tokenizes a single word using the learned merge ranks."""
        if word_tuple in word_cache:
            return word_cache[word_tuple]
            
        # Convert word list to a list of symbols
        parts = list(word_tuple)
        perc = 0
        while len(parts) > 1:
            # Find all current adjacent pairs in this word and look up their merge ranks
            pairs = [(parts[i], parts[i+1]) for i in range(len(parts) - 1)]
            
            # Find the pair that was trained earliest (lowest rank)
            # If none of the pairs exist in our merge rules, we are done merging this word
            valid_pairs = [p for p in pairs if p in merge_ranks]
            if not valid_pairs:
                break
                
            best_pair = min(valid_pairs, key=lambda p: merge_ranks[p])
            
            # Perform the merge for this specific best pair within the word
            new_parts = []
            i = 0
            first, second = best_pair
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == first and parts[i+1] == second:
                    new_parts.append(first + second)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        word_cache[word_tuple] = parts
        return parts

    # 2. Process all words (leveraging the cache automatically)
    token_ids = []
    for word in words:
        # Convert to tuple so it can be hashed/cached
        encoded_word = encode_word(tuple(word)) 
        
        for token in encoded_word:
            if token in vocab:
                token_ids.append(vocab[token])
            else:
                print(f"Warning: Character '{token}' not found in vocabulary. Skipping.")

    return token_ids