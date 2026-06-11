**How to use my Byte Pair Encoding-based tokenizer:**

1) import raw text data from a file or url
\n*in this example case (Hamlet by Shakespeare):
\n   from urllib.request import urlopen
\n   url = "https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt"
\n   with urlopen(url) as response:
\n     rawdata = response.read().decode('utf-8')*
\n
\n2) run trained_vocab, trained_merges = BPEtokenizer(rawdata)
\n*default target vocab size set to 5000, can be changed by adding a second parameter:
\nfor example: vocab, merges = BPEtokenizer(rawdata, 10000)
\nwill return a list of the trained tokens and merges*
\n
\n3) once training is complete, you can use the encode() and decode() functions to test
\n*encode returns a list of token ids calculated from the given text
\nexample:
\ntoken_ids = encode("To be, or not to be, that is the question.", trained_vocab, trained_merges)
\n--> returns [211, 122, 7, 276, 133, 99, 122, 7, 151, 131, 78, 1030, 9])
\n
\nfinal_text = decode(token_ids, trained_vocab)
\n--> returns "To be, or not to be, that is the question."
\n
\n4) this tokenizer can be used to tokenize/prepare any text for use in LLM projects, as each function is highly customizable and has its own parameters
