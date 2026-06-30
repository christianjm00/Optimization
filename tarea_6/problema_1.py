import numpy as np
# funcion beale
def fun_beale(x):
    return (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2

def grad_fun_beale(x):
    df_dx=2*(1.5-x[0]+x[0]*x[1])*(-1+x[1])+2*(2.25-x[0]+x[0]*x[1]**2)*(-1+x[1]**2)+2*(2.625-x[0]+x[0]*x[1]**3)*(-1+x[1]**3)

    df_dy=2*(1.5-x[0]+x[0]*x[1])*x[0]+2*(2.25-x[0]+x[0]*x[1]**2)*(2*x[1]*x[0])+2*(2.625-x[0]+x[0]*x[1]**3)*(3*x[0]*x[1]**2)
    return np.array([df_dx,df_dy])

def Hess_fun_beale(x):
    d2f_d2x=2*(-1+x[1])**2+2*(-1+x[1]**2)**2+2*(-1+x[1]**3)**2
    d2f_dxdy=2*x[0]*(-1+x[1])+2*(1.5-x[0]+x[0]*x[1])+4*x[0]*x[1]*(-1+x[1]**2)+4*x[1]*(2.25-x[0]+x[0]*x[1]**2)+6*x[0]*(x[1]**2)*(-1+x[1]**3)+6*(x[1]**2)*(2.625-x[0]+x[0]*x[1]**3)
    d2f_d2y=2*x[0]**2+8*(x[0]**2)*(x[1]**2)+18*(x[0]**2)*(x[1]**4)
    return np.array([[d2f_d2x,d2f_dxdy],[d2f_dxdy,d2f_d2y]])

#función himmelbau
def fun_Himmelbau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

def grad_fun_Himmelbau(x):
    df_dx=2*(x[0]**2+x[1]-11)*(2*x[0])+2*(x[0]+x[1]**2-7)
 

    df_dy=2*(x[0]**2+x[1]-11)+2*(x[0]+x[1]**2-7)*(2*x[1])
   
    return np.array([df_dx,df_dy])

def Hess_fun_Himmelbau(x):
    d2f_d2x=12*x[0]**2+4*x[1]-42
    d2f_dxdy=4*x[0]+4*x[1]
    d2f_d2y=12*x[1]**2+4*x[0]-26
    return np.array([[d2f_d2x,d2f_dxdy],[d2f_dxdy,d2f_d2y]])

# funcion hartmann
A=np.array([
    [  10,   3,   17, 3.5, 1.7,  8], 
    [0.05,  10,   17, 0.1,   8, 14],
    [   3, 3.5,  1.7,  10,  17,  8],
    [  17,   8, 0.05,  10, 0.1, 14]
])
P=10**(-4)*np.array([
[1312, 1696, 5569,  124, 8283, 5886],
[2329, 4135, 8307, 3736, 1004, 9991],
[2348, 1451, 3522, 2883, 3047, 6650],
[4047, 8828, 8732, 5743, 1091,  381]
])
alpha=np.array([1,1.2,3,3.2])

def exp_term(x, alpha,A,P):
    return alpha*np.exp(-np.sum(A*(x-P)**2,axis=1))

def fun_hartmann(x,alpha,A,P):
    ve=exp_term(x,alpha,A,P)
    return -(2.58+np.sum(ve))/1.94

def grad_fun_hartmann(x,alpha,A,P):
    ve=exp_term(x,alpha,A,P)
    grad=-np.sum(ve[:,np.newaxis]*(-2*A*(x-P)),axis=0)/1.94
    return grad

def hess_hartmann(x,alpha,A,P):
    ve = exp_term(x,alpha,A,P)
    dim = len(x)
    H = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            if i==j:
                H[i,j]=-np.sum(ve*((-2*A[:,i]*(x[i]-P[:,i]))**2-2*A[:,i]))/1.94
            else:
                H[i,j]=-np.sum(ve*(4*A[:,i]*A[:, j]*(x[i]-P[:,i])*(x[j]-P[:,j])))/1.94
    return H


# funcion rosenbrock
def fun_Rosenbrock(x):
    fu=np.sum(100*(x[1:]-x[:-1]**(2))**(2)+(1-x[:-1])**(2),axis=0)
    return fu

def gra_fun_Rosenbrock(x):
    dim=x.shape[0]
    ja_fu=np.zeros(dim)
    ja_fu[0]=-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0])
    ja_fu[1:-1]=200*(x[1:-1]-x[:-2]**2)-400*(x[2:]-x[1:-1]**2)*x[1:-1]-2*(1-x[1:-1])**2
    ja_fu[-1]=200*(x[-1]-x[-2]**2)
    return ja_fu

def Hess_fun_Rosenbrock(x):
    dim=x.shape[0]
    itera=np.arange(dim)
    He_fu=np.zeros((dim,dim))

    He_fu[itera[1:],itera[:-1]]=-400*x[:-1]
    He_fu[itera[:-1],itera[1:]]=-400*x[:-1]
    He_fu[itera[:-1],itera[:-1]]=-400*(x[1:]-x[:-1]**2)+800*x[:-1]**2+2
    He_fu[-1,-1]=200
    return He_fu


def cal_eigval(A):
    re=np.linalg.eigvalsh(A)
    return re[0],re[-1]

def class_eigval(eig_1,eig_n,imp=True):
    opcion=3
    if eig_1>0: 
        if imp:
            print("Matriz definida positiva")
        opcion=0
    elif eig_n<0:
        if imp:
            print(" Matriz definida negativa")
        opcion=1
    elif eig_1<0 and eig_n>0:
        if imp:
            print("Matriz indefinida")
        opcion=2
    return opcion

if __name__ == "__main__":
    print("evaluacion hess_fun_beale")
    x_beale_0=np.array([2,3])
    H_beale=Hess_fun_beale(x_beale_0)
    print(H_beale)
    eig_1,eig_n=cal_eigval(H_beale)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)


    print("evaluacion hess_fun_himmelbau")
    x_himme_0=np.array([1,2])
    H_himme=Hess_fun_Himmelbau(x_himme_0)
    print(H_himme)
    eig_1,eig_n=cal_eigval(H_himme)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)

    print("evaluacion hess_fun_hartmann")
    x_hart_0=np.array([0,0,0,0,0,0])
    H_hart=hess_hartmann(x_hart_0,alpha,A,P)
    print(H_hart)
    eig_1,eig_n=cal_eigval(H_hart)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)

    print("evaluacion hess_fun_rosenbrock")
    x_Rosen_0=np.array([-1.2,1])
    x_Rosen_0_1=np.ones(200)
    x_Rosen_0_1[0]=-1.2
    x_Rosen_0_1[-2]=-1.2
    x_Rosen_0_2=np.ones(600)
    x_Rosen_0_2[0]=-1.2
    x_Rosen_0_2[-2]=-1.2

    print("x_Rosen_0")
    H_Rosen=Hess_fun_Rosenbrock(x_Rosen_0)
    print(H_Rosen)
    eig_1,eig_n=cal_eigval(H_Rosen)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)

    print("x_rosen_0_1")
    H_Rosen=Hess_fun_Rosenbrock(x_Rosen_0_1)
    print(H_Rosen)
    eig_1,eig_n=cal_eigval(H_Rosen)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)

    print("x_rosen_0_2")
    H_Rosen=Hess_fun_Rosenbrock(x_Rosen_0_2)
    print(H_Rosen)
    eig_1,eig_n=cal_eigval(H_Rosen)
    print(eig_1,eig_n)
    class_eigval(eig_1,eig_n)



