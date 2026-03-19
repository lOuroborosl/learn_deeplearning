import numpy as np

def AND_gate(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    theta=-0.7
    res = False
    if np.sum(w * x) + theta > 0:
        res = True
    print( str(x1)+" AND "+ str(x2) +" is "+ str(res))
    return res

def NAND_gate(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    theta=0.7
    res = False
    if np.sum(w * x) + theta > 0:
        res = True
    print( str(x1)+" AND "+ str(x2) +" is "+ str(res))
    return res

def OR_gate(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    theta=-0.2
    res = False
    if np.sum(w * x) + theta > 0:
        res = True
    print( str(x1)+" AND "+ str(x2) +" is "+ str(res))
    return res

def XOR_gate(x1,x2):
    res = AND_gate(OR_gate(x1,x2),NAND_gate(x1,x2))
    print( str(x1)+" AND "+ str(x2) +" is "+ str(res))
    return res