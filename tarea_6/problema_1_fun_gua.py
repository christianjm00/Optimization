def exp_term(x,alpha,A,P):
    ve=alpha*np.exp(-np.dot(A*(x-P)**2))
    return  ve

def fun_hartmann(x,alpha,A,P):
    ve=exp_term(x,alpha,A,P)
    f=(2.58+np.sum(ve))/1.94
    return f 

def grad_fun_hartmann(x,alpha,A,P):
    ve=exp_term(x,alpha,A,P)
    oper=-2*np.dot(A,x-P)
    grad=-1*np.dot(ve,oper)/1.94
    return grad

def hess_hartmann(x,alpha,A,P):
    ve=exp_term(x,alpha,A,P)
    oper_0=-2*np.dot(A,x-P)
    oper_1=np.dot(oper_0.T,oper_0)
    H=-1*np.dot(ve,oper_1)/1.94
    i=np.arange(x)
    oper_2=oper_0**2-2*A
    H[i,i]=-np.dot(ve,oper_2)/1.94
    return H