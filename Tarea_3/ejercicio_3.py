import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time as tm
from Descenso_gradiente import Descenso_gradiente 
X = pd.read_csv("iris.csv").iloc[:,0:4].values
Specie = pd.read_csv("iris.csv")
pos_Setosa=Specie.index[Specie["variety"]=="Setosa"].tolist()
pos_Versicolor=Specie.index[Specie["variety"]=="Versicolor"].tolist()
pos_Virginica=Specie.index[Specie["variety"]=="Virginica"].tolist()

#print(X)
#print(Specie)
dim_X=X.shape
#print(dim_X)
Ma_Dij_x=np.zeros((dim_X[0],dim_X[0]))
#print(Ma_Dij_x.shape)

for i in range(len(X)-1):
    dif=X[i]-X[i+1:]
    dif_cua=dif**2
    sum_dif=np.sum(dif_cua,1)
    Ma_Dij_x[i,i+1:]=np.sqrt(sum_dif)
#print(Ma_Dij_x)

def fun_oj(z):
    dim=z.shape[0]
    z_re=z.reshape(dim//2,2)
    Ma_dij_z=np.zeros((dim_X[0],dim_X[0]))
    for i in range(len(z_re)-1):
        dif=z_re[i]-z_re[i+1:]
        dif_cua=dif**2
        sum_dif=np.sum(dif_cua,1)
        Ma_dij_z[i,i+1:]=np.sqrt(sum_dif)
    total=np.sum((Ma_Dij_x-Ma_dij_z)**2)
    return total

def gra_fun_oj(z):
    dim=z.shape[0]
    z_re=z.reshape(dim//2,2)
    Ma_oper_0=np.zeros((dim_X[0],dim_X[0],2))
    Ma_dij_z=np.zeros((dim_X[0],dim_X[0]))
    gra_fun=np.zeros((dim//2,2))
    for i in range(len(z_re)-1):
        dif=z_re[i]-z_re[i+1:]
        dif_cua=dif**2
        sum_dif=np.sum(dif_cua,1)
        dij_z=np.sqrt(sum_dif)
        oper_0=dif/(dij_z[:,np.newaxis])
        Ma_dij_z[i,i+1:]=np.sqrt(sum_dif)
        Ma_oper_0[i,i+1:,:]=oper_0
    
    for i in range(len(z_re)-1):
        oper_1=Ma_Dij_x[i,i+1:]-Ma_dij_z[i,i+1:]
        deriva=2*oper_1[:,np.newaxis]*Ma_oper_0[i,i+1:,:]
        gra_fun[i]=-np.sum(deriva,0)
        gra_fun[i+1:]=deriva
    #total=np.sum((Ma_dato-Ma_y)**2)
    return gra_fun.flatten()


def hess_fun_oj(z):
    dim = z.shape[0]//2 
    z_re = z.reshape(dim,2) 
    H = np.zeros((2*dim,2*dim)) 

    for i in range(dim):
        for j in range(i+1,dim): 
            dif=z_re[i]-z_re[j]
            dij=np.linalg.norm(dif)

            factor=(Ma_Dij_x[i,j]-dij)/dij
            outer=np.outer(dif, dif)/dij**3

            H_block=-2*(factor*np.eye(2)-factor*outer)
            H[2*i:2*i+2,2*j:2*j+2]=H_block
            H[2*j:2*j+2,2*i:2*i+2]=H_block

            H[2*i:2*i+2,2*i:2*i+2]-=H_block
            H[2*j:2*j+2,2*j:2*j+2]-=H_block

    return H
np.random.seed(1)
pun_ini=np.random.uniform(-1,4,300)
print("pun_ini:",pun_ini)
#print(fun_oj(pun_ini))
#print(gra_fun_oj(pun_ini))
#print(hess_fun_oj(pun_ini))

print("resultado: ")
print("Método 1: Alpha constante")
met_1=Descenso_gradiente(f=fun_oj,g=gra_fun_oj,H=hess_fun_oj,x_0=pun_ini,ty_alpha=1,paso=10**(-4),num_i=100000,error=10**(-8))
res_met_1=met_1.run()
print("iteraciones:",met_1.i)
print("x_n:",res_met_1[0])
print("f(x_n):",res_met_1[1])
print("error:",met_1.error)

print("Método 2: Alpha basado en la aproximación de Newton-Raphson.")
met_2=Descenso_gradiente(f=fun_oj,g=gra_fun_oj,H=hess_fun_oj,x_0=pun_ini,ty_alpha=2,paso=10**(-4),num_i=1000,error=10**(-8))
res_met_2=met_2.run()
print("iteraciones:",met_2.i)
print("x_n:",res_met_2[0])
print("f(x_n):",res_met_2[1])
print("error:",met_2.error)

print("Método 3: Alpha basado en una regla de ajuste iterativa.")
met_3=Descenso_gradiente(f=fun_oj,g=gra_fun_oj,H=hess_fun_oj,x_0=pun_ini,ty_alpha=3,paso=10**(-4),num_i=1000,error=10**(-8))
res_met_3=met_3.run()
print("iteraciones:",met_3.i)
print("x_n:",res_met_3[0])
print("f(x_n):",res_met_3[1])
print("error:",met_3.error)


x_n_met_1=met_1.x_n
x_n_met_1=x_n_met_1.reshape(150,2)
plt.figure()
plt.scatter(x_n_met_1[pos_Setosa,0],x_n_met_1[pos_Setosa,1],label="Setosa")
plt.scatter(x_n_met_1[pos_Versicolor,0],x_n_met_1[pos_Setosa,1],label="Versicolor")
plt.scatter(x_n_met_1[pos_Virginica,0],x_n_met_1[pos_Setosa,1],label="Virginica")
plt.title(r"Escalado multidimensional $\mathbb{R}^2$ con aprox. $\alpha$ de tipo 1")
plt.legend()
plt.savefig("Ta3_ej_3_gra_1.pdf")

x_n_met_2=met_2.x_n
x_n_met_2=x_n_met_2.reshape(150,2)
plt.figure()
plt.scatter(x_n_met_2[pos_Setosa,0],x_n_met_2[pos_Setosa,1],label="Setosa")
plt.scatter(x_n_met_2[pos_Versicolor,0],x_n_met_2[pos_Setosa,1],label="Versicolor")
plt.scatter(x_n_met_2[pos_Virginica,0],x_n_met_2[pos_Setosa,1],label="Virginica")
plt.title(r"Escalado multidimensional $\mathbb{R}^2$ con aprox. $\alpha$ de tipo 2")
plt.legend()
plt.savefig("Ta3_ej_3_gra_2.pdf")

x_n_met_3=met_3.x_n
x_n_met_3=x_n_met_3.reshape(150,2)
plt.figure()
plt.scatter(x_n_met_3[pos_Setosa,0],x_n_met_3[pos_Setosa,1],label="Setosa")
plt.scatter(x_n_met_3[pos_Versicolor,0],x_n_met_3[pos_Setosa,1],label="Versicolor")
plt.scatter(x_n_met_3[pos_Virginica,0],x_n_met_3[pos_Setosa,1],label="Virginica")
plt.title(r"Escalado multidimensional $\mathbb{R}^2$ con aprox. $\alpha$ de tipo 3")
plt.legend()
plt.savefig("Ta3_ej_3_gra_3.pdf")


