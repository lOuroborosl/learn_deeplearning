from config import config
import numpy as np

def AND_gate(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    theta=-0.7
    res = False
    if np.sum(w * x) + theta > 0:
        res = True
    print( str(x1)+" AND "+ str(x2) +" is "+ str(res))