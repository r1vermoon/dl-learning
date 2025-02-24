import numpy as np
import math

class value():
    def __init__(self, data, _child=(), _op='', _grad=0, label=''):
        self.data = data
        self.prev = _child
        self._op = _op
        self.grad = _grad
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value={self.data}"

    def __add__(self,other):
        
        other=other if isinstance(other,value) else value(other)
        out=value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = value(other)
        out = value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=value(t,(self, ),'tanh')

        def _backward():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backward
        
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float"
        x=self.data**other
        out=value(x,(self,),f'**{other}')

        def _backward():
            self.grad += other*(self.data**(other-1))*out.grad

        out._backward=_backward
        return out
        

    def exp(self):
        x=self.data
        out=value(math.exp(x),(self, ),'exp')

        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out

    def backward(self):
        nodes = list()

        def tree(v):
            if v not in nodes:
                nodes.append(v)
                for child in v.prev:
                    tree(child)

        tree(self)
        self.grad = 1.0
        for i in nodes:
            i._backward()


# forward
w1=value(-3,label='w1')
x1=value(2,label='x1')
w2=value(1,label='w2')
x2=value(0,label='x2')
b=value(6.8813735870195432,label='b')
w1x1=w1*x1;w1x1.label='w1x1'
w2x2=w2*x2;w2x2.label='w2x2'
w1x1w2x2=w1x1+w2x2;w1x1w2x2.label='w1x1w2x2'
n=w1x1w2x2+b;n.label='n'
e=n.exp();e.label='e'
e2=e.__pow__(2);e2.label='e2' # 这里backward有问题
c=e2+(-1);c.label='c'
d=e2+1;d.label='d'
d2=d.__pow__(-1);d2.label='d2'
o=c*d2;o.label='o'

o.backward()

print(n.grad)
