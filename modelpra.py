import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s: [stoi[c] for c in s]
decode=lambda l: ''.join([itos[i] for i in l])

data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

torch.manual_seed(1337)
batch_size=64
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4
eval_iters=200
n_embd=384
device='cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.2
n_head=6
n_layer=6

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x,y


class Head(nn.Module):

    def __init__(self, head_size) :
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)  
        q=self.query(x)
        wei=q@k.transpose(-2,-1)  #(B,T,16)@(B,16,T)-->(B,T,T)
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))  #mask效果
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei@v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size) :
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for  _ in range(num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))  
        return out
    
class FeedFoward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),  
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):

    def __init__(self, n_embd, n_head) :
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedFoward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=self.sa(self.ln1(x))
        x=self.ffwd(self.ln2(x))
        return x
    
class BigramlanguageModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)  
        self.blocks=nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),  
        )
        self.lm_head=nn.Linear(n_embd,vocab_size) 
    
    def forward(self,idx,targets=None):
        B,T=idx.shape
        
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device)) 
        x=tok_emb+pos_emb
        x=self.blocks(x)
        logits=self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:] 
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)  #(B,1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx




m=BigramlanguageModule()
optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)

for steps in range(100):
    xb,yb=get_batch('train')

    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=50)[0].tolist()))