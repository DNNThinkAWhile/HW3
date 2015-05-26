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
BATCH_SIZE = 7

real_mem_vector = shared( np.random.uniform( -1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3,(1, mem_vec_size)) )
error_grad = shared( np.zeros((GRAM_COUNT, BATCH_SIZE, y_size)))


#forward_mem_vector = shared(real_mem_vector.get_value())
                                                
w_i = shared( np.random.uniform(-1/np.sqrt(x_size)*3,       1/np.sqrt(x_size)*3,       (x_size, mem_vec_size)) )
w_h = shared( np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, mem_vec_size)) )
w_o = shared( np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, y_size)) )                               

input_x = T.dtensor3('input_x')
ans_y = T.dtensor3('ans_y')

#forward part
z_x = T.tensordot(input_x, w_i,1)

zh1 = T.dot(T.tile(real_mem_vector,(BATCH_SIZE,1)), w_h)
z1 = zh1 + z_x[0]
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


#grad
grad = 2*(ans_y - y)

#backpropagation
de_sig_z = y * (1 - y)
phi_y = y * (1 - y) * grad

sig_z_x = 1/(T.exp((-1)*z_x)+1)
de_sig_z_x = sig_z_x*(1-sig_z_x)

sig_zh = 1/(T.exp((-1)*zh)+1)
de_sig_zh = sig_zh*(1-sig_zh)

    #4
Wo_fix4 =  T.tensordot(a4.T, phi_y[3],(1,0))/ BATCH_SIZE
phi_i4 = theano.scan(lambda iter_de_sig_z_x4, iter_phi_y4: T.dot(iter_phi_y4 ,iter_de_sig_z_x4 * w_o.T), sequences = [de_sig_z_x[3],phi_y[3]])[0] 
Wi_fix_4_4 = T.sum(theano.scan(lambda iter_phi_i4, iter_x: T.tensordot(iter_x, iter_phi_i4,0), sequences =[phi_i4, input_x[3]])[0], axis = 0) / BATCH_SIZE
phi_h4, upd_phih4 = theano.scan(lambda iter_de_sig_zh4, iter_phi_y4: T.dot(iter_phi_y4 ,iter_de_sig_zh4 * w_o.T), sequences = [de_sig_zh[3],phi_y[3]])
Wh_fix_4_4 = T.sum(theano.scan(lambda iter_phi_h4, iter_a3: T.tensordot(iter_a3, iter_phi_h4,0), sequences =[phi_h4, a[2]])[0], axis = 0) / BATCH_SIZE

phi_i3, upd_phii3 = theano.scan(lambda iter_de_sig_z_x3, iter_phi_h4: T.dot(iter_phi_h4, iter_de_sig_z_x3 * w_h.T), sequences = [de_sig_z_x[2],phi_h4]) 
Wi_fix_4_3 = T.sum(theano.scan(lambda iter_phi_i3, iter_x: T.tensordot(iter_x, iter_phi_i3,0), sequences =[phi_i3, input_x[2]])[0], axis = 0) / BATCH_SIZE
phi_h3, upd_phih3 = theano.scan(lambda iter_de_sig_zh3, iter_phi_h4: T.dot(iter_phi_h4, iter_de_sig_zh3 * w_h.T), sequences = [de_sig_zh[2],phi_h4])
Wh_fix_4_3 = T.sum(theano.scan(lambda iter_phi_h3, iter_a2: T.tensordot(iter_a2, iter_phi_h3,0), sequences =[phi_h3, a[1]])[0], axis = 0) / BATCH_SIZE

phi_i2, upd_phii2 = theano.scan(lambda iter_de_sig_z_x2, iter_phi_h3: T.dot(iter_phi_h3, iter_de_sig_z_x2 * w_h.T), sequences = [de_sig_z_x[1],phi_h3]) 
Wi_fix_4_2 = T.sum(theano.scan(lambda iter_phi_i2, iter_x: T.tensordot(iter_x, iter_phi_i2,0), sequences =[phi_i2, input_x[1]])[0], axis = 0) / BATCH_SIZE
phi_h2, upd_phih2 = theano.scan(lambda iter_de_sig_zh2, iter_phi_h3: T.dot(iter_phi_h3, iter_de_sig_zh2 * w_h.T), sequences = [de_sig_zh[1],phi_h3])
Wh_fix_4_2 = T.sum(theano.scan(lambda iter_phi_h2, iter_a1: T.tensordot(iter_a1.T, iter_phi_h2,0), sequences =[phi_h2, a[0]])[0], axis = 0) / BATCH_SIZE

phi_i1, upd_phii1 = theano.scan(lambda iter_de_sig_z_x1, iter_phi_h2: T.dot(iter_phi_h2, iter_de_sig_z_x1 * w_h.T), sequences = [de_sig_z_x[0],phi_h2]) 
Wi_fix_4_1 = T.sum(theano.scan(lambda iter_phi_i1, iter_x: T.tensordot(iter_x, iter_phi_i1,0), sequences =[phi_i1, input_x[0]])[0], axis = 0) / BATCH_SIZE
phi_h1, upd_phih1 = theano.scan(lambda iter_de_sig_zh1, iter_phi_h2: T.dot(iter_phi_h2, iter_de_sig_zh1 * w_h.T), sequences = [de_sig_zh[0],phi_h2])
Wh_fix_4_1 = T.sum(theano.scan(lambda iter_phi_h1: T.tensordot(real_mem_vector, iter_phi_h1,0), sequences = phi_h1)[0], axis = 0)[0] / BATCH_SIZE

    #3
Wo_fix3 =  T.tensordot(a3.T, phi_y[2],(1,0)) / BATCH_SIZE
phi_i3, upd_phii3 = theano.scan(lambda iter_de_sig_z_x3, iter_phi_y3: T.dot(iter_phi_y3, iter_de_sig_z_x3 * w_o.T), sequences = [de_sig_z_x[2],phi_y[2]]) 
Wi_fix_3_3 = T.sum(theano.scan(lambda iter_phi_i3, iter_x: T.tensordot(iter_x, iter_phi_i3,0), sequences =[phi_i3, input_x[2]])[0], axis = 0) / BATCH_SIZE
phi_h3, upd_phih3 = theano.scan(lambda iter_de_sig_zh3, iter_phi_y3: T.dot(iter_phi_y3, iter_de_sig_zh3 * w_o.T), sequences = [de_sig_zh[2],phi_y[2]])
Wh_fix_3_3 = T.sum(theano.scan(lambda iter_phi_h3, iter_a3: T.tensordot(iter_a3, iter_phi_h3,0), sequences =[phi_h3, a[1]])[0], axis = 0) / BATCH_SIZE

phi_i2, upd_phii2 = theano.scan(lambda iter_de_sig_z_x2, iter_phi_h3: T.dot(iter_phi_h3, iter_de_sig_z_x2 * w_h.T), sequences = [de_sig_z_x[1],phi_h3]) 
Wi_fix_3_2 = T.sum(theano.scan(lambda iter_phi_i2, iter_x: T.tensordot(iter_x, iter_phi_i2,0), sequences =[phi_i2, input_x[1]])[0], axis = 0) / BATCH_SIZE
phi_h2, upd_phih2 = theano.scan(lambda iter_de_sig_zh2, iter_phi_h3: T.dot(iter_phi_h3, iter_de_sig_zh2 * w_h.T), sequences = [de_sig_zh[1],phi_h3])
Wh_fix_3_2 = T.sum(theano.scan(lambda iter_phi_h2, iter_a1: T.tensordot(iter_a1, iter_phi_h2,0), sequences =[phi_h2, a[0]])[0], axis = 0) / BATCH_SIZE

phi_i1, upd_phii1 = theano.scan(lambda iter_de_sig_z_x1, iter_phi_h2: T.dot(iter_phi_h2, iter_de_sig_z_x1 * w_h.T), sequences = [de_sig_z_x[0],phi_h2]) 
Wi_fix_3_1 = T.sum(theano.scan(lambda iter_phi_i1, iter_x: T.tensordot(iter_x, iter_phi_i1,0), sequences =[phi_i1, input_x[0]])[0], axis = 0) / BATCH_SIZE
phi_h1, upd_phih1 = theano.scan(lambda iter_de_sig_zh1, iter_phi_h2: T.dot(iter_phi_h2, iter_de_sig_zh1 * w_h.T), sequences = [de_sig_zh[0],phi_h2])
Wh_fix_3_1 = T.sum(theano.scan(lambda iter_phi_h1: T.tensordot(real_mem_vector, iter_phi_h1,0), sequences = phi_h1)[0], axis = 0)[0] / BATCH_SIZE

    #2
Wo_fix2 =  T.tensordot(a2.T, phi_y[1],(1,0)) / BATCH_SIZE
phi_i2, upd_phii2 = theano.scan(lambda iter_de_sig_z_x2, iter_phi_y2: T.dot(iter_phi_y2, iter_de_sig_z_x2 * w_o.T), sequences = [de_sig_z_x[1],phi_y[1]]) 
Wi_fix_2_2 = T.sum(theano.scan(lambda iter_phi_i2, iter_x: T.tensordot(iter_x, iter_phi_i2,0), sequences =[phi_i2, input_x[1]])[0], axis = 0) / BATCH_SIZE
phi_h2, upd_phih2 = theano.scan(lambda iter_de_sig_zh2, iter_phi_y2: T.dot(iter_phi_y2, iter_de_sig_zh2 * w_o.T), sequences = [de_sig_zh[1],phi_y[1]])
Wh_fix_2_2 = T.sum(theano.scan(lambda iter_phi_h2, iter_a2: T.tensordot(iter_a2, iter_phi_h2,0), sequences =[phi_h2, a[0]])[0], axis = 0) / BATCH_SIZE

phi_i1, upd_phii1 = theano.scan(lambda iter_de_sig_z_x1, iter_phi_h2: T.dot(iter_phi_h2, iter_de_sig_z_x1 * w_h.T), sequences = [de_sig_z_x[0],phi_h2]) 
Wi_fix_2_1 = T.sum(theano.scan(lambda iter_phi_i1, iter_x: T.tensordot(iter_x, iter_phi_i1,0), sequences =[phi_i1, input_x[0]])[0], axis = 0) / BATCH_SIZE
phi_h1, upd_phih1 = theano.scan(lambda iter_de_sig_zh1, iter_phi_h2: T.dot(iter_phi_h2,iter_de_sig_zh1 * w_h.T), sequences = [de_sig_zh[0],phi_h2])
Wh_fix_2_1 = T.sum(theano.scan(lambda iter_phi_h1: T.tensordot(real_mem_vector, iter_phi_h1,0), sequences = phi_h1)[0], axis = 0)[0] / BATCH_SIZE

    #1
Wo_fix1 =  T.tensordot(a1.T, phi_y[0],(1,0)) / BATCH_SIZE
phi_i1, upd_phii1 = theano.scan(lambda iter_de_sig_z_x1, iter_phi_y1: T.dot(iter_phi_y1, iter_de_sig_z_x1 * w_o.T), sequences = [de_sig_z_x[0],phi_y[0]]) 
Wi_fix_1_1 = T.sum(theano.scan(lambda iter_phi_i1, iter_x: T.tensordot(iter_x, iter_phi_i1,0), sequences =[phi_i1, input_x[0]])[0], axis = 0) / BATCH_SIZE
phi_h1, upd_phih1 = theano.scan(lambda iter_de_sig_zh1, iter_phi_y1: T.dot(iter_phi_y1, iter_de_sig_zh1 * w_o.T), sequences = [de_sig_zh[0],phi_y[0]])
Wh_fix_1_1 = T.sum(theano.scan(lambda iter_phi_h1: T.tensordot(real_mem_vector, iter_phi_h1,0), sequences = phi_h1)[0], axis = 0)[0] / BATCH_SIZE 


#update
update_w_o = w_o - Wo_fix1 - Wo_fix2 - Wo_fix3 - Wo_fix4
update_w_i = w_i-Wi_fix_1_1-Wi_fix_2_1-Wi_fix_2_2-Wi_fix_3_1-Wi_fix_3_2-Wi_fix_3_3-Wi_fix_4_1-Wi_fix_4_2-Wi_fix_4_3-Wi_fix_4_4
update_w_h = w_h-Wh_fix_1_1-Wh_fix_2_1-Wh_fix_2_2-Wh_fix_3_1-Wh_fix_3_2-Wh_fix_3_3-Wh_fix_4_1-Wh_fix_4_2-Wh_fix_4_3-Wh_fix_4_4


forward = function ( [input_x, ans_y], \
                     [theano.Out(real_mem_vector, borrow= True), theano.Out(w_i, borrow = True), theano.Out(w_h, borrow = True), \
                      theano.Out(w_o ,borrow = True), theano.Out(error_grad, borrow = True)], \
                     updates = [(error_grad, grad),(w_o, update_w_o), (w_i, update_w_i), (w_h, update_w_h)])
#                     [theano.Out(Wh_fix_3_3, borrow= True), theano.Out(Wh_fix_4_1, borrow = True), theano.Out(Wh_fix_4_2, borrow = True), \
#                      theano.Out(Wh_fix_4_3 ,borrow = True), theano.Out(Wh_fix_4_4, borrow = True)]) 
                     
def init(x_size, mem_vec_size, y_size):
    w_i = np.random.uniform(-1/np.sqrt(x_size)*3, 1/np.sqrt(x_size)*3, (x_size, mem_vec_size))
    w_h = np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, mem_vec_size))
    w_o = np.random.uniform(-1/np.sqrt(mem_vec_size)*3, 1/np.sqrt(mem_vec_size)*3, (mem_vec_size, y_size))
    return w_i, w_h, w_o
    
def main():
    x = np.random.uniform(-1,1,(GRAM_COUNT, BATCH_SIZE, x_size))
    ans_y = np.zeros((GRAM_COUNT, BATCH_SIZE, y_size))    
    
    for i in range(GRAM_COUNT):
        for j in range(BATCH_SIZE):
            k = np.random.randint(0,y_size)
            ans_y[i][j][k] = 1

    re1,re2,re3,re4,re5 = forward(x, ans_y)
#    print np.shape(re1) , ',' , np.shape(w_i), ',' , np.shape(w_h), ',' , np.shape(w_o), ',' , np.shape(error_grad)
    print np.shape(re1) , ',' , np.shape(re2), ',' , np.shape(re3), ',' , np.shape(re4), ',' , np.shape(re5)
    print 'success!!!!!!!!!!!!!!!!!!!'
    

if __name__ == '__main__':
   main()
