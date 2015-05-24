import numpy as np
from theano import function
import theano.tensor as T

Output = T.dvector('Output')
OutputG = Output*(1-Output)
SigmoidGrad = function([Output],OutputG, mode='FAST_RUN')

EleVector1 = T.dvector('EleVector1')
EleVector2 = T.dvector('EleVector2')
EleWiseVV = function([EleVector1, EleVector2], EleVector1*EleVector2, mode='FAST_RUN')

DotMatrix1 = T.dmatrix('DotMatrix1')
DotMatrix2 = T.dmatrix('DotMatrix2')
MatrixDot = function([DotMatrix1, DotMatrix2], T.dot(DotMatrix1, DotMatrix2), mode='FAST_RUN')

DotMatrix3 = T.dmatrix('DotMatrix3')
DotVector1 = T.dvector('DotVector1')
MatrixVecDot = function([DotMatrix3, DotVector1], T.dot(DotMatrix3, DotVector1), mode='FAST_RUN')

MultiVec1 = T.dvector('MultiVec1')
MultiMatrix1 = T.dmatrix('MultiMatrix1')
VecMatMulti = function([MultiVec1, MultiMatrix1], MultiVec1*MultiMatrix1, mode='FAST_RUN')

def VecToMatrixAndDot (vec1, vec2):
    return MatrixDot(np.matrix(vec1).T, np.matrix(vec2))

def SigZWPhi (z, w, phi):
    return MatrixVecDot((VecMatMulti(SigmoidGrad(z),w.T)).T,phi)

def CheckExploid (w_delta, w_th, w_pre):
    return (w_delta, np.absolute(w_delta).sum()) if np.absolute(w_delta).sum() < 100 * w_th else (-1 *w_pre, np.absolute(w_pre).sum())

def BPTT_Theano(W, C, Z, A):
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
    Wfixtmp = 99999
    Wpre = 0
    for l in range(len(C) - 1, -1, -1):
        phi = 0
        for i in range(l - 1, -1, -1):
            if i == l - 1:
                # Wo part
                phi = EleWiseVV(SigmoidGrad(Zo[i]), C[l])
                Wpre, Wfixtmp = CheckExploid(VecToMatrixAndDot(Ao[i], phi), Wfixtmp, Wpre)
                Wo_fix.append(Wpre)
                # Wi part
                phi_i = SigZWPhi(Zi[i], Wo, phi)
                Wpre, Wfixtmp = CheckExploid(VecToMatrixAndDot(Ai[i], phi_i), Wfixtmp, Wpre)
                Wi_fix.append(Wpre)
                # Wh part
                phi = SigZWPhi(Zh[i], Wo, phi)
                Wpre, Wfixtmp = CheckExploid(VecToMatrixAndDot(Ah[i], phi), Wfixtmp, Wpre)
                Wh_fix.append(Wpre)
            else:
                # Wi part
                phi_i = SigZWPhi(Zi[i], Wh, phi)
                Wpre, Wfixtmp = CheckExploid(VecToMatrixAndDot(Ai[i], phi_i), Wfixtmp, Wpre)
                Wi_fix.append(Wpre)
                # Wh part
                phi = SigZWPhi(Zh[i], Wh, phi)
                Wpre, Wfixtmp = CheckExploid(VecToMatrixAndDot(Ah[i], phi), Wfixtmp, Wpre)
                Wh_fix.append(Wpre)
    # Fix all Ws
    for fix in Wo_fix:
        Wo = Wo - fix
    for fix in Wi_fix:
        Wi = Wi - fix
    for fix in Wh_fix:
        Wh = Wh - fix
    return (Wo, Wi, Wh)
	
def main():
    Wi = np.ones((3,2)) * 0.5
    Wh = np.ones((2,2)) * 0.3
    Wo = np.ones((2,5)) * 0.7
    W = (Wo, Wi, Wh)
    C = [None] * 3
    C[0] = np.array([0.3,0.2,0.5,0.7,0.11])
    C[1] = np.array([0.78,0.235,0.32,0.235,0.932])
    C[2] = np.array([0.43,0.12,0.45,0.75,0.2711])
    Ai = [None] * 3
    Ai[0] = np.array([0.1,0.2,0.3])
    Ai[1] = Ai[0] *0.5
    Ai[2] = Ai[0] * 2 
    Ao = [None] * 3
    Ao[0] = np.array([0.5,0.2])
    Ao[1] = np.array([0.5,0.2])
    Ao[2] = np.array([0.5,0.2])
    Ah = [None] * 3
    Ah[0] = np.array([0.3,0.6])
    Ah[1] = np.array([0.3,0.6])
    Ah[2] = np.array([0.3,0.6])
    A = (Ao, Ai, Ah)
    Zi = [None] * 3
    Zi[0] = np.array([0.34,0.76])
    Zi[1] = np.array([0.34,0.76])
    Zi[2] = np.array([0.34,0.76])
    Zo = [None] * 3
    Zo[0] = np.array([0.3,0.543,0.12,0.74,0.23])
    Zo[1] = np.array([0.3,0.543,0.12,0.74,0.23])
    Zo[2] = np.array([0.3,0.543,0.12,0.74,0.23])
    Zh = [None] * 3
    Zh[0] = np.array([0.23,0.43])
    Zh[1] = np.array([0.23,0.43])
    Zh[2] = np.array([0.23,0.43])
    Z = (Zo, Zi, Zh)
    print 'Wo = ',BPTT_Theano(W, C, Z, A)[0]
    print 'Wi = ',BPTT_Theano(W,C,Z,A)[1]
    print 'Wh = ',BPTT_Theano(W,C,Z,A)[2]

if __name__ == "__main__":
    main()
