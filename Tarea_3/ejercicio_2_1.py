import numpy as np
from Descenso_gradiente import Descenso_gradiente
def fun_Rosenbrock(x):
    fu = np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
    return fu
def gra_fun_Rosenbrock(x):
    dim = x.shape[0]
    ja_fu = np.zeros(dim)
    ja_fu[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    ja_fu[1:-1] = 200 * (x[1:-1] - x[:-2]**2) - 400 * x[1:-1] * (x[2:] - x[1:-1]**2) - 2 * (1 - x[1:-1])
    ja_fu[-1] = 200 * (x[-1] - x[-2]**2)
    return ja_fu
def He_fun_Rosenbrock(x):
    dim = x.shape[0]
    itera = np.arange(dim)
    He_fu = np.zeros((dim, dim))

    He_fu[itera[1:], itera[:-1]] = -400 * x[:-1]
    He_fu[itera[:-1], itera[1:]] = -400 * x[:-1]
    He_fu[itera[:-1], itera[:-1]] = 800 * x[:-1]**2 - 400 * (x[1:] - x[:-1]**2) + 2
    He_fu[-1, -1] = 200
    return He_fu


punto_inicial=np.ones(128)
punto_inicial[0]=-1.2
punto_inicial[-2]=-1.2

print(fun_Rosenbrock(punto_inicial))
print(gra_fun_Rosenbrock(punto_inicial))
print(He_fun_Rosenbrock(punto_inicial))
print("resultado: ")
print("Método 1: Alpha constante")
alpha_met_1=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=1,paso=0.0005,num_i=10000000,error=10**(-20))
res_alpha_met_1=alpha_met_1.run()
print("iteraciones:",alpha_met_1.i)
print("x_n:",res_alpha_met_1[0])
print("f(x_n):",res_alpha_met_1[1])

print("Método 2: Alpha basado en la aproximación de Newton-Raphson.")
alpha_met_2=Descenso_gradiente(f=fun_Rosenbrock,g=gra_fun_Rosenbrock,H=He_fun_Rosenbrock,x_0=punto_inicial,ty_alpha=2,paso=0.0005,num_i=10000000,error=10**(-20))
res_alpha_met_2=alpha_met_2.run()
print("iteraciones:",alpha_met_2.i)
print("x_n:",res_alpha_met_2[0])
print("f(x_n):",res_alpha_met_2[1])
print("Método 3: Alpha basado en una regla de ajuste iterativa.")