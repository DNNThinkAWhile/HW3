import numpy as np

def np_sig (np_array):
    return 1/(np.exp((-1)*np_array+1))

def np_sig_grad (np_array):
    sig = np_sig(np_array)
    return sig * (1 - sig)

def np_dot (mat1, mat2):
    return np.dot(np.matrix(mat1).T, np.matrix(mat2))


# Weight
# W = (Wo(numpy), Wi(numpy), Wh(numpy)), Error(list)
# Input
# C(numpy), Z = (Zo(list), Zi(list), Zh(list)), A = (Ao(list), Ai(list), Ah(list)) !!! Warning: Ai SHOULD AND MUST BE X !!! 
# Data
# X(list), Init(Numpy)
def BPTT (W, Error, C, Z, A, X, Init):
    # De-tuple
    Wo = W[0]
    Wi = W[1]
    Wh = W[2]
    Zo = Z[0]
    Zi = Z[1]
    Zh = Z[2]
    Ao = A[0]
    Ai = A[1]
    Ah = A[2]
    #
    Wo_fix = []
    Wi_fix = []
    Wh_fix = []
    for l in range(len(Error) - 1, -1, -1):
        phi = 0
        for i in range(l - 1, -1, -1):
            if i == l - 1:
                # Wo part
                phi = np_sig_grad(Zo[i])*C
                Wo_fix.append( np_dot(phi, Ao[i]) )
                # Wi part
                phi_i = np_sig_grad(Zi[i])*(Wo.T)*phi
                Wi_fix.append( np_dot(phi_i, Ai[i]) )
                # Wh part
                phi = np_sig_grad(Zh[i])*(Wo.T)*phi
                Wh_fix.append( np_dot(phi, Ah[i]) )
            else:
                # Wi part
                phi_i = np_sig_grad(Zi[i])*(Wh.T)*phi
                Wi_fix.append( np_dot(phi_i), Ai[i] )
                # Wh part
                phi = np_sig_grad(Zh[i])*(Wh.T)*phi
                Wh_fix.append( np_dot(phi, Ah[i]) )
    # Fix all Ws
    for fix in Wo_fix:
        Wo = Wo - fix
    for fix in Wi_fix:
        Wi = Wi - fix
    for fix in Wh_fix:
        Wh = Wh - fix
    return (Wo, Wi, Wh)


def main():
    BPTT()


if "__init__" == "__main__":
    main()
