import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano import shared
import random

x_size = 5
y_size = 15
mem_vec_size =10 
GRAM_COUNT = 4
BATCH_SIZE = 128

real_mem_vector = shared( np.random.uniform( -1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3,(1, mem_vec_size)) )

#forward_mem_vector = shared(real_mem_vector.get_value())
                                                
w_i = shared( np.random.uniform(-1/np.sqrt(x_size)*3,       1/np.sqrt(x_size)*3,       (x_size, mem_vec_size)) )
w_h = shared( np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, mem_vec_size)) )
w_o = shared( np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, y_size)) )                               

input_x = T.dtensor3('input_x')
#ans_y = T.dtensor3('ans_y')

z_x = T.tensordot(input_x, w_i,1)

z1 = T.dot(T.tile(real_mem_vector,(BATCH_SIZE,1)), w_h) + z_x[0]
a1 = 1/(T.exp((-1)*z1)+1)

z2 = T.dot(a1, w_h) + z_x[1]
a2 = 1/(T.exp((-1)*z2)+1)

z3 = T.dot(a2, w_h) + z_x[2]
a3 = 1/(T.exp((-1)*z3)+1)

z4 = T.dot(a3, w_h) + z_x[3]
a4 = 1/(T.exp((-1)*z4)+1) 

a = T.concatenate([a1, a2, a3, a4], axis = 0)
z = T.concatenate([z1, z2, z3, z4], axis = 0)

zy = T.tensordot(a,w_o,1)
y =  1/(T.exp((-1)*zy)+1)


forward = function ( [input_x], [theano.Out(y,borrow = True), theano.Out(a,borrow = True), theano.Out(z,borrow = True)])                                

def init(x_size, mem_vec_size, y_size):        
    w_i = np.random.uniform(-1/np.sqrt(x_size)*3, 1/np.sqrt(x_size)*3, (x_size, mem_vec_size))
    w_h = np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, mem_vec_size))
    w_o = np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, y_size))
    return w_i, w_h, w_o

def main():
    x = np.random.uniform(-1,1,(GRAM_COUNT, BATCH_SIZE, x_size))
    ans_y = np.zeros((GRAM_COUNT, BATCH_SIZE, x_size))    
    
    for i in range(GRAM_COUNT):
        for j in range(BATCH_SIZE):
            k = np.random.randint(0,x_size)
            ans_y[i][j][k] = 1
           
    print w_i.get_value()
    y,a,z = forward(x)
    print 'success!!!!!!!!!!!!!!!!!!!'
    

if __name__ == '__main__':
   main()
