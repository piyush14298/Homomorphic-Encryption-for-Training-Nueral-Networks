import numpy as np

def generate_private_key(w,row,col):
    K = (np.random.rand(row,col) * w / (2 ** 16)) # proving max(S) < w
    return K

def encryption(p,K,row,col,w):    
    e = (np.random.rand(row)) # proving max(e) < w / 2
    c = np.linalg.inv(K).dot((w * p) + e)
    return c

def decryption(c,K,w):
    return (K.dot(c) / w).astype('int')

def compute_cstar(c,row,u):
    cstar = np.zeros(u * row,dtype='int')
    for i in range(row):
        b = np.array(list(np.binary_repr(np.abs(c[i]))),dtype='int')
        if(c[i] < 0):
            b *= -1
        cstar[(i * u) + (u-len(b)): (i+1) * u] += b
    return cstar

def switch_key(c,K,row,col,T):
    u = int(np.ceil(np.log2(np.max(np.abs(c)))))
    cstar = compute_cstar(c,row,u)
    Kstar = compute_Kstar(K,row,col,u)
    colprime = col + 1
    

    Kprime = np.concatenate((np.eye(row),T.T),0).T
    A = (np.random.rand(colprime - row, col*u) * 10).astype('int')
    E = (1 * np.random.rand(Kstar.shape[0],Kstar.shape[1])).astype('int')
    M = np.concatenate(((Kstar - T.dot(A) + E),A),0)
    cprime = M.dot(cstar)
    return cprime,Kprime

def compute_Kstar(K,row,col,u):
    Kstar = list()
    for i in range(u):
        Kstar.append(K*2**(u-i-1))
    Kstar = np.array(Kstar).transpose(1,2,0).reshape(row,col*u)
    return Kstar

def get_T(col):
    colprime = col + 1
    T = (10 * np.random.rand(col,colprime - col)).astype('int')
    return T

def encrypt_using_switchKey(p,w,row,col,T):
    c,K = switch_key(p*w,np.eye(row),row,col,T)
    return c,K

p = np.array([0,1,2,5])

row = len(p)
col = row
w = 16
K = generate_private_key(w, row, col)
print("Plain Text is: "+str(p))
print("\nPrivate Key is: "+str(K))
T = get_T(col)
c = encryption(p, K, row, col, w)
print("\nCipher Text is: " + str(c))
print("\nAfter Decryption: "+str(decryption(c, K, w)))
print("\nAfter addition then Decryption: "+str(decryption(c+c, K, w)))
print("\nAfter multiplication then Decryption: "+str(decryption(c*5, K, w)))