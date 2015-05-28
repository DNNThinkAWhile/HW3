import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared
import random

x_size = 100
y_size = 53606 
mem_vec_size = 10 
GRAM_COUNT = 4
BATCH_SIZE = 20

model_w_i = np.random.uniform(-1,1,(x_size, mem_vec_size))
model_w_h = np.random.uniform(-1,1,(mem_vec_size, mem_vec_size))
model_w_o = np.random.uniform(-1,1,(mem_vec_size, y_size))


real_mem_vector = shared(np.zeros((1,mem_vec_size)))

w_i = shared( model_w_i )
w_h = shared( model_w_h )
w_o = shared( model_w_o )                               

input_x = T.dtensor3('input_x')

z_x = T.tensordot(input_x, w_i,1)

zh1 = T.dot(T.tile(real_mem_vector,(BATCH_SIZE,1)), w_h)
z1 = z_x[0]
a1 = 1/(T.exp((-1)*z1)+1)

zh2 = T.dot(a1, w_h)
z2 = zh2 + z_x[1]
a2 = 1/(T.exp((-1)*z2)+1)

zh3 = T.dot(a2, w_h)
z3 = zh3 + z_x[2]
a3 = 1/(T.exp((-1)*z3)+1)

zh4 = T.dot(a3, w_h)
z4 = zh4 + z_x[3]
a4 = 1/(T.exp((-1)*z4)+1) 

a = T.stacklists([a1, a2, a3, a4])
z = T.stacklists([z1, z2, z3, z4])
zh = T.stacklists([zh1, zh2, zh3, zh4])
zy = T.tensordot(a,w_o,1)
y =  1/(T.exp((-1)*zy)+1)

forward = function ( [input_x],theano.Out(y, borrow= True))

def main():
    x = np.random.uniform(-1,1,(GRAM_COUNT, BATCH_SIZE, x_size))

    re1= forward(x)
#    print np.shape(re1) , ',' , np.shape(w_i), ',' , np.shape(w_h), ',' , np.shape(w_o), ',' , np.shape(error_grad)
    print np.shape(re1) 
    print 'success!!!!!!!!!!!!!!!!!!!'
    

if __name__ == '__main__':
    main()


