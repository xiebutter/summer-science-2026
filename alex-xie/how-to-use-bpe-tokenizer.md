**How to use my Byte Pair Encoding-based tokenizer:**

1) import raw text data from a file or url
 *in this example case (Hamlet by Shakespeare):
   from urllib.request import urlopen
   url = "https://gist.githubusercontent.com/provpup/2fc41686eab7400b796b/raw/b575bd01a58494dfddc1d6429ef0167e709abf9b/hamlet.txt"
   with urlopen(url) as response:
     rawdata = response.read().decode('utf-8')*

2) run trained_vocab, trained_merges = BPEtokenizer(rawdata)
*default target vocab size set to 5000, can be changed by adding a second parameter:
for example: vocab, merges = BPEtokenizer(rawdata, 10000)
will return a list of the trained tokens and merges*

3) once training is complete, you can use the encode() and decode() functions to test
*encode returns a list of token ids calculated from the given text
example:
token_ids = encode("To be, or not to be, that is the question.", trained_vocab, trained_merges)
--> returns [211, 122, 7, 276, 133, 99, 122, 7, 151, 131, 78, 1030, 9])

final_text = decode(token_ids, trained_vocab)
--> returns "To be, or not to be, that is the question."

4) this tokenizer can be used to tokenize/prepare any text for use in LLM projects, as each function is highly customizable and has its own parameters
