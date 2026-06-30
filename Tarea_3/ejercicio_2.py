import numpy as np
from Descenso_gradiente import Descenso_gradiente
import matplotlib.pyplot as plt
def fun_Rosenbrock(x):
    fu=np.sum(100*(x[1:]-x[:-1]**(2))**(2)+(1-x[:-1])**(2))
    return fu

def gra_fun_Rosenbrock(x):
    dim=x.shape[0]
    ja_fu=np.zeros(dim)
    ja_fu[0]=-400*(x[1]-x[0]**2)*x[0]-2*(1-x[0])
    ja_fu[1:-1]=200*(x[1:-1]-x[:-2]**2)-400*(x[2:]-x[1:-1]**2)*x[1:-1]-2*(1-x[1:-1])**2
    ja_fu[-1]=200*(x[-1]-x[-2]**2)
    return ja_fu

def He_fun_Rosenbrock(x):
    dim=x.shape[0]
    itera=np.arange(dim)
    He_fu=np.zeros((dim,dim))

    He_fu[itera[1:],itera[:-1]]=-400*x[:-1]
    He_fu[itera[:-1],itera[1:]]=-400*x[:-1]
    He_fu[itera[:-1],itera[:-1]]=-400*(x[1:]-x[:-1]**2)+800*x[:-1]**2+2
    He_fu[-1,-1]=200
    return He_fu

punto_inicial=np.ones(128)
punto_inicial[0]=-1.2
punto_inicial[-2]=-1.2

print("evaluaciones con un punto dado")
print("resultado: ")

print("Método 1: Alpha constante")
met_1_1=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=1,paso=10**(-4),num_i=100000,error=10**(-7))
res_met_1_1=met_1_1.run()
print("iteraciones:",met_1_1.i)
print("x_n:",res_met_1_1[0])
print("f(x_n):",res_met_1_1[1])
print("error:",met_1_1.error)

print("Método 2: Alpha basado en la aproximación de Newton-Raphson.")
met_2_1=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=2,paso=10**(-4),num_i=100000,error=10**(-7))
res_met_2_1=met_2_1.run()
print("iteraciones:",met_2_1.i)
print("x_n:",res_met_2_1[0])
print("f(x_n):",res_met_2_1[1])
print("error:",met_2_1.error)

print("Método 3: Alpha basado en una regla de ajuste iterativa.")
met_3_1=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=3,paso=10**(-4),num_i=100000,error=10**(-7))
res_met_3_1=met_3_1.run()
print("iteraciones:",met_3_1.i)
print("x_n:",res_met_3_1[0])
print("f(x_n):",res_met_3_1[1])
print("error:",met_3_1.error)

print("evaluaciones con un punto random")
np.random.seed(42)
punto_inicial=np.random.uniform(-1.5,1.5,128)
print("resultado: ")
print("Método 1: Alpha constante")
met_1_2=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=1,paso=10**(-4),num_i=100000,error=10**(-5))
res_met_1_2=met_1_2.run()
print("iteraciones:",met_1_2.i)
print("x_n:",res_met_1_2[0])
print("f(x_n):",res_met_1_2[1])
print("error:",met_1_2.error)

print("Método 2: Alpha basado en la aproximación de Newton-Raphson.")
met_2_2=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=2,paso=10**(-4),num_i=100000,error=10**(-5))
res_met_2_2=met_2_2.run()
print("iteraciones:",met_2_2.i)
print("x_n:",res_met_2_2[0])
print("f(x_n):",res_met_2_2[1])
print("error:",met_2_2.error)

print("Método 3: Alpha basado en una regla de ajuste iterativa.")
met_3_2=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=3,paso=10**(-4),num_i=100000,error=10**(-5))
res_met_3_2=met_3_2.run()
print("iteraciones:",met_3_2.i)
print("x_n:",res_met_3_2[0])
print("f(x_n):",res_met_3_2[1])
print("error:",met_3_2.error)

plt.figure()
plt.yscale('log')
plt.plot(np.arange(met_1_1.i+2),met_1_1.memor_f,marker='o', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_1")
plt.plot(np.arange(met_2_1.i+2),met_2_1.memor_f,marker='s', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_2")
plt.plot(np.arange(met_3_1.i+2),met_3_1.memor_f,marker='d', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_3")
plt.legend()
plt.title("Optimización de la función de Rosenbrock evaluada,con punto inicio dado "+r"$x_0=[-1.2,1,1,\dots,1,-1.2]$")
plt.xlabel("iteraciones k")
plt.ylabel(r"$f(x_k)$")
plt.savefig("Ta3_ej_2_gra_1.pdf")

plt.figure()
plt.yscale('log')
plt.plot(np.arange(met_1_1.i+1),np.linalg.norm(met_1_1.memor_grad,axis=1),marker='o', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_1")
plt.plot(np.arange(met_2_1.i+1),np.linalg.norm(met_2_1.memor_grad,axis=1),marker='s', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_2")
plt.plot(np.arange(met_3_1.i+1),np.linalg.norm(met_3_1.memor_grad,axis=1),marker='d', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_3")
plt.legend()
plt.title("Gradiente de la función Rosenbrock evaluada, con punto inicio dado "+r"$x_0=[-1.2,1,1,\dots,1,-1.2]$")
plt.xlabel("iteraciones k")
plt.ylabel(r"$f(x_k)$")
plt.savefig("Ta3_ej_2_gra_2.pdf")


plt.figure()
plt.yscale('log')
plt.plot(np.arange(met_1_1.i+2),met_1_1.memor_f,marker='o', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_1")
plt.plot(np.arange(met_2_1.i+2),met_2_1.memor_f,marker='s', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_2")
plt.plot(np.arange(met_3_1.i+2),met_3_1.memor_f,marker='d', linestyle='-',markevery=100,label=r"$f(x_k)$ con el met_3")
plt.legend()
plt.title("Optimización de la función de Rosenbrock evaluada,con punto inicio aleatorio")
plt.xlabel("iteraciones k")
plt.ylabel(r"$f(x_k)$")
plt.savefig("Ta3_ej_2_gra_3.pdf")

plt.figure()
plt.yscale('log')
plt.plot(np.arange(met_1_1.i+1),np.linalg.norm(met_1_1.memor_grad,axis=1),marker='o', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_1")
plt.plot(np.arange(met_2_1.i+1),np.linalg.norm(met_2_1.memor_grad,axis=1),marker='s', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_2")
plt.plot(np.arange(met_3_1.i+1),np.linalg.norm(met_3_1.memor_grad,axis=1),marker='d', linestyle='-',markevery=100,label=r"$\nabla f(x_k)$ con el met_3")
plt.legend()
plt.title("Gradiente de la función Rosenbrock evaluada, con punto inicio aleatorio ")
plt.xlabel("iteraciones k")
plt.ylabel(r"$f(x_k)$")
plt.savefig("Ta3_ej_2_gra_4.pdf")