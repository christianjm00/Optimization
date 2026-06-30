import numpy as np
import pickle

class Descenso_gradiente:
    """
    Descenso del gradiente \n
    información: \n
    Entrada: \n
    f: funcion \n
    g: funcion \n
    H: funcion \n
    x_0: punto incial \n
    ty_approx: int 1,2 y 3 \n
    error: \n
    Variables ocultas:

    Salida: \n
    """
    def __init__(self,f,g,H,x_0,ty_alpha,paso,error=10**(-5),num_i=100):
        self.f=f
        self.g=g
        self.H=H
        self.x_0=x_0
        self.error=error
        self.num_i=num_i
        self.x_n=None
        self.alpha=None
        self.i=0
        self.ty_alpha=ty_alpha
        self.paso=paso
        self.memor_error=[]
        self.memor_x=[]
        self.memor_f=[]
        if ty_alpha==1:
            self.fun_alpha=self.alpha_ty_1
        elif ty_alpha==2:
            self.fun_alpha=self.alpha_ty_2
        elif ty_alpha==3:      
            self.fun_alpha=self.alpha_ty_3
            
    def alpha_ty_1(self):
        """paso= peso de avance"""
        self.alpha=self.paso

    def alpha_ty_2(self):
        g_x = self.g(self.memor_x[-1])  
        H_x = self.H(self.memor_x[-1])

        # Asegúrate de que el Hessiano no sea singular (si es posible, agregar una pequeña regularización)
        den=np.dot(g_x, np.dot(H_x, g_x))
        if den != 0:
            self.alpha = np.dot(g_x, g_x) / den
        else:
            self.alpha = self.paso  # Si el Hessiano es singular, usar un paso constante

    def alpha_ty_3(self):
        if self.i == 0:
            self.alpha = self.paso
        else:
            g_x = self.g(self.memor_x[-1])
            # Asegúrate de que la diferencia entre f_i y f_ip1 sea significativa
            if self.memor_f[-2] != self.memor_f[-1]:
                self.alpha = (self.alpha**2) * np.dot(g_x, g_x) / (2 * (self.memor_f[-2] - self.memor_f[-1] + self.alpha * np.dot(g_x, g_x)))
            else:
                self.alpha = self.paso  # Evitar división por cero si las funciones son muy similares

    def run(self):
        x_i = self.x_0
        f_i = self.f(x_i)
        self.memor_x.append(x_i)
        self.memor_f.append(f_i)
        for self.i in range(self.num_i):
            self.fun_alpha()
            x_ip1 = x_i - self.alpha * self.g(x_i)
            f_ip1 = self.f(x_ip1)
            error_i = np.abs(f_ip1 - f_i)
            if error_i < self.error:
                break
            self.memor_error.append(error_i)
            self.memor_x.append(x_ip1)
            self.memor_f.append(f_ip1)
            x_i = x_ip1
        self.x_n = x_ip1  
        return x_ip1

    def guardar(self, name):
        with open(name + ".pkl", 'wb') as archivo:
            pickle.dump(self, archivo)

    @classmethod
    def cargar(csl, name):
        with open(name + ".pkl", 'rb') as archivo:
            i_c = pickle.load(archivo)

        return i_c

if __name__ == "__main__":
    def fun(x):
        return np.array(x**2 + 5*x + 7)
    
    def gra(x):
        return np.array(2*x + 5)
    
    def Hes(x):
        return np.array([[2]])  # Matriz Hessiana de dimensión 1x1
    
    prueba = Descenso_gradiente(f=fun, g=gra, H=Hes, x_0=np.array([1]), ty_alpha=2, paso=0.5, num_i=1000)
    resultado = prueba.run()
    print(resultado)
    prueba.guardar("prueba")

