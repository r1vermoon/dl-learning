import torch
import torch.nn.functional as F

words=open('names.txt','r').read().splitlines()
#print(words[:2])

#itos,stoi
chars=sorted(list(set(''.join(words))))
#print(chars)
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
#print(itos[3])

#构建概率矩阵
X,Y=[],[]
for w in words:
    chs=['.']+list(w)+['.']
    for x,y in zip(chs,chs[1:]):
        ix=stoi[x]
        iy=stoi[y]
        X.append(ix)
        Y.append(iy)
N=torch.zeros((27,27),dtype=torch.int32)
for x,y in zip(X,Y):
    N[x,y]+=1
#print(N[0].sum())
N=N/N.sum(dim=1,keepdim=True)
#print(N[0])

#likelihood
likelihood=0
n=0

for w in words:
    chs=['.']+list(w)+['.']
    for x,y in zip(chs,chs[1:]):
        ix=stoi[x]
        iy=stoi[y]
        prob=N[ix,iy]
        logprob=torch.log(prob)
        likelihood+=logprob
        n+=1

likelihood=-1*likelihood
likelihood=likelihood/n
print(likelihood)

#generate
g=torch.Generator().manual_seed(123456789)
for i in range(5):
    W=[]  #注意，在for里面
    ix=0
    while True:
        p=N[ix]
        ix=torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        W.append(itos[ix])
        if ix==0: 
            break
    print(''.join(W))


#network
W=torch.randn((27,27),generator=g,requires_grad=True)
xs,ys=[],[]

#构建神经网络
for w in words:
    chs=['.']+list(w)+['.']
    for x,y in zip(chs,chs[1:]):
        ix=stoi[x]
        iy=stoi[y]
        xs.append(ix)
        ys.append(iy)

#backward
xs=torch.tensor(xs)
ys=torch.tensor(ys)
num=xs.nelement()

for i in range(50):
    xenc=F.one_hot(xs,num_classes=27).float()
    out = xenc@W
    out=out.exp()/out.exp().sum(dim=1,keepdim=True)
    #print(out[0].sum())
    loss=-out[torch.arange(num),ys].log().mean()
    print(loss.item())
    
    W.grad=None
    loss.backward()
    W.data += -100 * W.grad

#generate

for i in range(5):

    D=[]
    ix=0
    while True:
        xenc=F.one_hot(torch.tensor(ix),num_classes=27).float()
        out=xenc @ W
        out=out.exp()
        probs=out/out.sum(0,keepdim=True)
        ix=torch.multinomial(probs,num_samples=1,replacement=True,generator=g).item()
        D.append(itos[ix])
        if ix==0:
            break
    print(''.join(D))

