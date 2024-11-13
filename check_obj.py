import numpy as np
    
def check_obj(max = None,sol = None): 
    B = np.arange(1,max+1)
    a = (np.round(sol))
    A = np.unique(a)
    if len(A) != len(a):
        C = np.setdiff1d(B, A, assume_unique=False)
        m = 1
        for k in np.arange(len(a)).reshape(-1):
            # n = np.find(a == a(k))
            n = np.where(a == a[k])
            if len(n) != 0:
                for j in np.arange(2,len(n)+1).reshape(-1):
                    a[n[j]] = C[m]
                    m = m + 1
    sol = a
    return sol
