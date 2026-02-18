with open('inputdata/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print ("length of dataset in characters: ", len(text))

#print (text[:1000])  # print the first 1000 characters of the text

chars = sorted(list(set(text)))

vocab_size = len(chars)
print ("number of unique characters: ", vocab_size)
print (''.join(chars))  # print the unique characters

char_to_idx = { ch:i for i,ch in enumerate(chars) }
idx_to_char = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [char_to_idx[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([idx_to_char[i] for i in l]) # decoder: take a list of integers, output a string

stringExample = "hii there. This is amazing. So are you"
print (encode(stringExample))  # encode a string into a list of integers
print (decode(encode(stringExample)))  # decode the list of integers back into a string


import torch
data = torch.tensor(encode(text), dtype=torch.long)
# print (data.shape, data.dtype)  # shape: (number of characters in the text,)
# print (data[:1000])  # print the first 1000 encoded characters as integers  
n = (int)( 0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#print (len(train_data), len(val_data))

block_size = 8
x = train_data[:block_size]

print (x)  # input sequence of length block_size
y = train_data[1:block_size+1]  # target sequence is the input sequence
for t in range(block_size):
    context = x[:t+1]  # the context is everything up to and including the current time step
    target = y[t]  # the target is the next character after the context
    #print(f"when input is {context} the target: {target}")  

torch.manual_seed(1337)  # set the random seed for reproducibility
batch_size = 4  # how many independent sequences will we process in parallel?   
block_size = 8  # what is the maximum context length for predictions?
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
xb, yb = get_batch('train')
# print("inputs:")
# print(xb)
# print("targets:")
# print(yb)

# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target: {target.item()}")  


import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print("logits shape: ", logits.shape)  # should be (B*T, C
print("loss: ", loss)

print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))