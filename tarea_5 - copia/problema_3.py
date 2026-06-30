import numpy as np
from problema_1 import Descenso_gradiente,maxima_verosimilitud,gra_maxima_verosimilitud,fun_logistica
from problema_2 import error
import gzip, pickle
import matplotlib.pyplot as plt
with gzip.open('mnist.pkl.gz','rb') as ff :
    u = pickle._Unpickler( ff )
    u.encoding = 'latin1'
    train, val, test = u.load()

train_x=train[0]
val_x=val[0]
test_x=test[0]

train_y=train[1]
#print(train_y)
val_y=val[1]
test_y=test[1]

dim_train_x=train_x.shape
dim_val_x=val_x.shape
dim_test_x=test_x.shape

dim_train_y=train_y.shape
dim_val_y=val_y.shape
dim_test_y=test_y.shape

def sele_cord(dim,datos_y):
    cord_ge=np.arange(dim)
    true_fa=(datos_y==0)|(datos_y==1)
    cord=cord_ge[true_fa]
    return cord

cord_train=sele_cord(dim_train_x[0],train_y)
cord_val=sele_cord(dim_val_x[0],val_y)
cord_test=sele_cord(dim_test_x[0],test_y)

train_x_01=train_x[cord_train]
val_x_01=val_x[cord_val]
test_x_01=test_x[cord_test]

train_y_01=train_y[cord_train].reshape(-1,1)
val_y_01=val_y[cord_val].reshape(-1,1)
test_y_01=test_y[cord_test].reshape(-1,1)
print("train_y_01:",train_y_01)
print("train_y_01",train_y_01.shape)

def fun_ob(var):
    pi=fun_logistica(var,1,train_x_01)
    return maxima_verosimilitud(pi,train_y_01)

def gra_fun_ob(var):
    pi=fun_logistica(var,1,train_x_01)
    return gra_maxima_verosimilitud(pi,train_x_01,train_y_01)
np.random.seed(1)
x_0=np.random.uniform(-2,2,dim_train_x[1]+1)
print("ejecutando descenso gradiente con metodo simple")
sol_simple=Descenso_gradiente(f=fun_ob,g=gra_fun_ob,H=None,x_0=x_0,dat_X_train=train_x_01,dat_y_train=train_y_01,tam_lot=None,met_algor="SIMPLE",paso=10**(-4),error=10**(-8),num_i=100)
resul_simp=sol_simple.run()
sol_simple.guardar("metodo_simple")
#print("sol_simple :",sol_simple.)
print("sol_simple i:",sol_simple.i)
print("sol_simple error:",sol_simple.error)
#print("resul_simp",resul_simp[0])
print("errores en metodo simple")
pi_simp=fun_logistica(resul_simp[0],1,train_x_01)
print("error de acierto con datos train",error(pi_simp,train_y_01))

pi_simp=fun_logistica(resul_simp[0],1,val_x_01)
print("error de acierto con datos val",error(pi_simp,val_y_01))

pi_simp=fun_logistica(resul_simp[0],1,test_x_01)
print("error de acierto con datos test",error(pi_simp,test_y_01))

print("ejecutando descenso gradiente con metodo de backtracking")
sol_back=Descenso_gradiente(f=fun_ob,g=gra_fun_ob,H=None,x_0=x_0,dat_X_train=train_x_01,dat_y_train=train_y_01,tam_lot=None,met_algor="backtracking",paso=10**(-4),error=10**(-8),num_i=100)
resul_back=sol_back.run()
sol_back.guardar("metodo_back")
#print(resul_back)
print("sol_back i:",sol_back.i)
print("sol_back error:",sol_back.error)
#print("resul",resul_back)

print("errores en metodo backtracking")
pi_back=fun_logistica(resul_back[0],1,train_x_01)
print("error de acierto con datos train",error(pi_back,train_y_01))

pi_back=fun_logistica(resul_back[0],1,val_x_01)
print("error de acierto con datos val",error(pi_back,val_y_01))

pi_back=fun_logistica(resul_back[0],1,test_x_01)
print("error de acierto con datos test",error(pi_back,test_y_01))

plt.figure()
plt.yscale('log')
plt.plot(np.arange(sol_simple.i+1),sol_simple.memor_error,label="Simple")
plt.plot(np.arange(sol_back.i+1),sol_back.memor_error,label="backtracking")
plt.title("Gráfica de los métodos, iteraciones (k) vs. " + r"$\|g_k\|$")
plt.xlabel("k (iteraciones)")
plt.xlabel(r"$\|g_k\|$")
plt.savefig("gra_probl_3_1.pdf")

plt.figure()
plt.yscale('log')
plt.plot(np.arange(sol_simple.i+2),sol_simple.memor_f,label="Simple")
plt.plot(np.arange(sol_back.i+2),sol_back.memor_f,label="backtracking")
plt.title("Gráfica de los métodos, iteraciones (k) vs"+ r"$-h_k$")
plt.xlabel("k (iteraciones)")
plt.xlabel(r"$-h_k$")
plt.savefig("gra_probl_3_2.pdf")
plt.show()