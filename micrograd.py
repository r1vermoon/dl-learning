import torch
import math
import numpy
import random
from typing import Any


class Value:
    def __init__(self,data,_children=(),op=''):
        self.data=data
        self.grad=0.0
        self._backward=lambda:None
        self._prev=set(_children)
        self.op=op
        self.grad=0.0

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self,other):
        
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out
    
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out
    
    def __rmul__(self,other):
        return self*other
    
    def __radd__(self,other):
        return self+other
    
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self, ),'exp')

        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float"
        x=self.data**other
        out=Value(x,(self,),f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad

        out._backward=_backward
        return out
    
    def __truediv__(self,other):
        return self*other**-1
    
    def __neg__(self):
        return self*-1

    def __sub__(self,other):
        return self + (-other)
    
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self, ),'tanh')

        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        
        return out
    
    def backward(self):
        
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad=1.0
        for node in reversed(topo):
            node._backward()

    
class Neuron:

    def __init__(self,nin) :
        self.w=[(Value(random.uniform(-1,1)))for _ in range(nin)] #随机生成，执行nin次,uniform浮点型
        self.b=Value(random.uniform(-1,1))
    
    def __call__(self, x): 
        #可被实例调用
        #print (list(zip(self.w,x)))
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out=act.tanh()
        return out
    
    def parameters(self):
        return self.w+[self.b]
    
class Layer:
    def __init__(self,nin,nout) :
        self.neurons=[Neuron(nin) for _ in range(nout)]  #生成节点对应w和b

    def __call__(self, x) :
        outs=[n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self,nin,nouts) :
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]



x=[2.0,3.0,-1.0]
n=MLP(3,[4,4,1])  #3个输入节点，后面三层的节点数分别为4，4，1
n(x)  #调用__call__()

xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
ys=[1.0,
    -1.0,
    -1.0,
    1.0]

for k in range(20):
    ypred=[n(x) for x in xs]
    loss= sum(
            [(yout-ygt)**2 for yout,ygt in zip(ypred,ys)]
        )
    # loss=Value(sum([((yout-ygt)**2).data for ygt,yout in zip(ys,ypred)]))

    for p in n.parameters():
        p.grad=0.0
    loss.backward()

    for p in n.parameters():
        p.data += -0.001*p.grad
    
    print(k,loss.data)


    