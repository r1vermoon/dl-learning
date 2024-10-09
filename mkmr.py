import torch
import torch.nn.functional as F

N=torch.zeros((27,27),dtype=torch.int32)
words=open('names.txt','r').read().splitlines()

chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1=stoi[ch1]
        ix2=stoi[ch2]
        N[ix1,ix2]+=1

P=(N+1).float()
P=P/P.sum(1,keepdim=True)

g=torch.Generator().manual_seed(2147483647)
for i in range(5):

    out=[]
    ix=0
    while True:
        p = P[ix]
        #p = p / p.sum()
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))

log_likelihood=0.0
n=0

for w in words:
    chs=['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob=P[ix1,ix2]
        logprob=torch.log(prob)
        log_likelihood += logprob
        n += 1

nll=-log_likelihood
nll=nll/n
print(f'{nll}')
print(f'{log_likelihood}')


xs,ys = [],[]

for w in words:
    chs=['.']+ list(w) +['.']
    for ch1,ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs=torch.tensor(xs)
num=xs.nelement()
ys=torch.tensor(ys)
g=torch.Generator().manual_seed(2147483647)
W=torch.rand((27,27),generator=g,requires_grad=True)

for k in range(100):
    xenc=F.one_hot(xs,num_classes=27).float()
    logits=xenc @ W
    counts=logits.exp()
    probs=counts/counts.sum(1,keepdim=True)
    loss=-probs[torch.arange(num),ys].log().mean()
    print(loss.item())

    W.grad=None
    loss.backward()

    W.data += -100 * W.grad


g=torch.Generator().manual_seed(2147483647)
for i in range(5):

    out=[]
    ix=0
    while True:
        xenc=F.one_hot(torch.tensor(ix),num_classes=27).float()
        logits=xenc @ W
        counts=logits.exp()
        probs=counts/counts.sum(0,keepdim=True)
        ix=torch.multinomial(probs,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix==0:
            break
    print(''.join(out))