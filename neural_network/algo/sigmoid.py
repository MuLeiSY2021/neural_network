import math

from algo.matrix import Matrix

def sigmoid(x) :
    
    if isinstance(x,(float,Matrix)):
        return 1/(1+ math.e**(-x))
    return NotImplemented

def mean_abs(m):
    r = len(m)
    c = len(m[0])
    s = 0.0
    for i in range(r):
        row = m[i]
        for j in range(c):
            s += abs(row[j])
    return s / (r * c)