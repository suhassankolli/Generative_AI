import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transfomer_blocks import  Block


#print("Torch version: ", torch.__version__)
#print("CUDA available: ", torch.cuda.is_available())
#print("GPU name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [ss+ "<END>" for ss in corpus]
text = " ".join(corpus)

#print(text)

words = list(set(text.split(" ")))

#print(words)

vocab_size = len(words)

#print(vocab_size)

word2index = {ss: ii for ii, ss in enumerate(words)}
print("word2index: ", word2index)

idx2word = {ii: ss for ss, ii in word2index.items()}
print("--------------------------------")
print("idx2word: ", idx2word)


data = torch.tensor([word2index[w] for w in text.split(" ")] , dtype=torch.long)


print("data: ", data)   
print("--------------------------------")
print(len(data))


block_size = 6
embedding_dim = 32
n_heads = 2
n_layers = 2
lr=le-3
epochs=1500