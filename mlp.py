import torch
import torch.nn.functional as F

words=open('names.txt','r').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
block_size = 3

X,Y=[],[]
for w in words:
    context = [0] * block_size
    for ch in w + '.' :
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(itos[i] for i in context), '--->',itos[ix])
        context = context[1:] + [ix]

X=torch.Tensor(X)
X=X.to(torch.int64)
Y=torch.Tensor(Y)
Y=Y.to(torch.int64)

g=torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2),generator=g)
W1=torch.rand((6,100),generator=g)
b1=torch.rand((100),generator=g)
W2=torch.randn((100,27),generator=g)
b2=torch.randn((27),generator=g)
parameters=[C,W1,b1,W2,b2]

for p in parameters:
    p.requires_grad=True

for i in range(50):

    #ix=torch.randint(0,X.shape[0],(32,))
    emb=C[X]
    h=torch.tanh(emb.view(-1,6)@W1 + b1)
    logits=h @ W2 + b2
    #counts=logits.exp()
    #probs=counts/counts.sum(1,keepdim=True)
    loss=F.cross_entropy(logits,Y)

    for p in parameters:
        p.grad=None
    loss.backward()

    lr=0.1
    for p in parameters:
        p.data += -lr * p.grad

    print(loss.item())
