import numpy as np
import scipy.linalg as spla
import sys
import ctypes as ct 
      

def pintamat(m,n,A):
    for i in range(m):
        for j in range(n):
            print ("%24.16e "%(A[i][j])),
        print("")

qr_givens = ct.cdll.LoadLibrary('./qr_givens.o')
qr_givens.qrgivens.argtypes=[ct.c_int, ct.c_int, ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.POINTER(ct.c_double))]
     
m=5; n=5;
A=np.random.random((m,n));
Q,R=spla.qr(A)
print(Q,'\n')
print(R,'\n')


print(A)

pA=(ct.POINTER(ct.c_double)*m)()
for i in range(0,m):
    pA[i]=(ct.c_double*n)()
    for j in range(0,n):
        pA[i][j]=A[i,j]

pQ=(ct.POINTER(ct.c_double)*m)()
for i in range(0,m):
    pQ[i]=(ct.c_double*n)()


print ("Anem a cridar la funcio qr_givens:" )
qr_givens.qrgivens(ct.c_int(m),ct.c_int(n),pA,pQ)

pintamat(m,n,pQ)
print("")
pintamat(m,n,pA)
