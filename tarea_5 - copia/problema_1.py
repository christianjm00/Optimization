import numpy as np

import numpy as np
import pickle
import inspect
class Descenso_gradiente:
    """
    Implementación del método de descenso del gradiente.
    ----------------------------------------------------
    Parámetros de entrada:
    - f: Función objetivo a minimizar.
    - g: Gradiente de la función f.
    - H: Matriz Hessiana de f.
    - x_0: Punto inicial del descenso (array).
    - ty_alpha: Método para calcular el paso alpha (1, 2 o 3).
    - paso: Valor inicial o constante para alpha.
    - error: Tolerancia del criterio de parada.
    - num_i: Número máximo de iteraciones.

    Variables internas:
    - x_n: Última iteración del descenso (solución encontrada).
    - alpha: Último valor utilizado para alpha.
    - i: Número de iteraciones realizadas.
    - memor_error: Historial de errores por iteración.
    - memor_x: Historial de los puntos evaluados.
    - memor_f: Historial de los valores de la función evaluados.
    - memor_gra: Historial de los valores del gradiente de la función evaluados
    """

    def __init__(self,f,g,H,x_0,dat_X_train,dat_y_train,tam_lot,met_algor,paso,error=10**(-5),num_i=100):
        self.f=f
        self.f_eva=None
        self.g=g
        self.g_eva=None
        self.H=H
        self.H_eva=None
        self.x_0=x_0
        self.error=error
        self.num_i=num_i
        self.cal_cam=0
        self.x_n=None
        self.dat_X_train=dat_X_train
        self.dat_y_train=dat_y_train
        self.tam_lot=tam_lot
        self.alpha=None
        self.met_algor=met_algor
        self.i=0
        self.paso=paso
        self.fun_cal=None
        self.memor_error=[]
        self.memor_x=[]
        self.memor_f=[]
        self.memor_grad=[]
        self.fun_met_algo=[]
        if self.met_algor=="SIMPLE":
            self.fun_alpha=self.alpha_ty_1
            self.fun_cal=self.simple
        elif self.met_algor=="backtracking":
            self.fun_cal=self.backtracking
        
    
    def batch_generador(self):
        """
        Genera lotes aleatorios de datos de tamaño batch_size.
        """
        idx=np.random.permutation(len(self.dat_X_train))  # Indices aleatorios
        X_shuffled=self.dat_X_train[idx]  # Reordenar X
        y_shuffled=self.dat_y_train[idx]  # Reordenar y
        pun_div= np.arange(self.tam_lot, len(self.dat_X_train), self.tam_lot)
        X_shuffled=np.split(X_shuffled,pun_div)
        y_shuffled=np.split(y_shuffled,pun_div)
        # Generar lotes
        return X_shuffled,y_shuffled

    def simple(self):
        self.fun_alpha()
        self.cal_cam=-self.alpha*self.g_eva

    def alpha_ty_1(self):
        """Método 1: Alpha constante (valor fijo determinado por 'paso')."""
        self.alpha=self.paso
    
    def backtracking(self):
        """
        Método de búsqueda de línea con bisección y condiciones de Wolfe.
        """
        if self.i==0:
            self.alpha=self.paso  # Valor inicial de alpha
        else:
            self.alpha=0  # Siguiendo la imagen
            self.beta=np.inf  # Límite superior
            self.alpha_i=self.paso  # Valor inicial de búsqueda
            self.c_1=0.5  # Condición de Armijo
            self.c_2=0.9  # Condición de curvatura
            
            while True:
                # Calcular nuevo punto
                x_ip1=self.x_i-self.alpha_i*self.g_eva

                # Evaluar función y gradiente en el nuevo punto
                caller=inspect.stack()[1].function
                if caller=="run":
                    f_ip1=self.f(x_ip1)
                    g_ip1=self.g(x_ip1)
                elif caller=="run_es":
                    f_ip1=self.f(x_ip1, self.dat_x_batch)
                    g_ip1=self.g(x_ip1, self.dat_x_batch, self.dat_y_batch)

                # Evaluar condición de Armijo
                eval_0=self.f_i-self.c_1*self.alpha_i*np.dot(self.g_eva, self.g_eva)
                if f_ip1>eval_0:
                    self.beta=self.alpha_i
                    self.alpha_i=0.5*(self.alpha+self.beta)

                # Evaluar condición de curvatura
                elif np.dot(g_ip1,-self.g_eva)<self.c_2*np.dot(self.g_eva,-self.g_eva):
                    self.alpha=self.alpha_i
                    self.alpha_i=2*self.alpha if self.beta==np.inf else 0.5*(self.alpha+self.beta)
                else:
                    break  # Condiciones de Wolfe satisfechas
            self.alpha=self.alpha_i  # Guardar el valor final
        self.cal_cam=-self.alpha*self.g_eva  # Actualizar dirección
    def run(self):
        """
        Ejecuta el algoritmo de descenso del gradiente.

        Devuelve:
        - Último punto encontrado (x_n), minimizador de f.
        - mínimo de f.
        """
        self.x_i=self.x_0
        self.f_i=self.f(self.x_i)
        self.memor_x.append(self.x_i)
        self.memor_f.append(self.f_i)
        for self.i in range(self.num_i):
            self.g_eva=self.g(self.x_i)
            self.fun_cal()
            self.x_ip1=self.x_i+self.cal_cam
            self.f_ip1=self.f(self.x_ip1)
            error_i=np.linalg.norm(self.g_eva)
            self.memor_error.append(error_i)
            self.memor_x.append(self.x_ip1)
            self.memor_f.append(self.f_ip1)
            self.memor_grad.append(self.g_eva)
            if error_i<=self.error:
                break
            
            self.x_i=self.x_ip1
            self.f_i=self.f_ip1
            print("iteracion:",self.i)
        self.x_n=self.x_ip1  
        return self.x_ip1,self.f_i
    
    def run_es(self):
        """
        Ejecuta el algoritmo de descenso del gradiente utilizando minibatches.

        Parámetros:
        - batch_size: Tamaño de los lotes para el entrenamiento.
        """
        self.x_i=self.x_0
        cal_batch=self.batch_generador()
        self.dat_x_batch=cal_batch[0][0]
        self.dat_y_batch=cal_batch[1][0]
        self.f_i=self.f(self.x_i,self.dat_x_batch)
        self.memor_x.append(self.x_i)
        self.memor_f.append(self.f_i)
        intera=1
        for self.i in range(self.num_i):
            # Generar lotes aleatorios
            #self.dat_x_batch,self.dat_y_batch=self.batch_generador_1()
            
            self.g_eva=self.g(self.x_i,self.dat_x_batch,self.dat_y_batch)  # Evaluar el gradiente para el batch
            self.fun_cal()  # Calcular la actualización del gradiente
            self.x_ip1=self.x_i+self.cal_cam  # Actualización de los parámetros
            self.f_ip1=self.f(self.x_ip1,self.dat_x_batch)
            error_i=np.linalg.norm(self.g_eva)
            self.memor_error.append(error_i)
            self.memor_x.append(self.x_ip1)
            self.memor_f.append(self.f_ip1)
            self.memor_grad.append(self.g_eva)
            # Verificar condición de parada por error
            if error_i<=self.error:
                break

            self.x_i=self.x_ip1
            self.f_i=self.f_ip1
            
            if intera <=(len(cal_batch[0])-1):
                self.dat_x_batch=cal_batch[0][intera]
                self.dat_y_batch=cal_batch[1][intera]
            else:
                cal_batch=self.batch_generador()
                intera=-1
            intera+=1

        self.x_n=self.x_ip1
        return self.x_ip1,self.f_ip1

    def guardar(self,name):
        """
        Guarda el estado actual del objeto en un archivo pickle.

        Parámetros:
        - name: Nombre del archivo sin extensión.
        """
        with open(name+".pkl", 'wb') as archivo:
            pickle.dump(self, archivo)

    @classmethod
    def cargar(csl,name):
        """
        Carga un objeto Descenso_gradiente desde un archivo pickle.

        Parámetros:
        - name: Nombre del archivo sin extensión.

        Devuelve:
        - Objeto cargado de tipo Descenso_gradiente.
        """
        with open(name+".pkl", 'rb') as archivo:
            i_c = pickle.load(archivo)

        return i_c


def fun_logistica(var,num_lo,x):
    ma_theta=var[:-num_lo].reshape((num_lo,-1))
    ve_sesgos=var[-num_lo:]
    z=np.dot(ma_theta,x.T)+ve_sesgos[:,np.newaxis]
    fun=1/(1+np.exp(-z))
    return fun.T

def maxima_verosimilitud(pi_i,y):
    return -1*np.sum(y*np.log(pi_i+10**(-6))+(1-y)*np.log(1-pi_i+10**(-6)))

def gra_maxima_verosimilitud(pi, x, y):
    x_mod=np.column_stack((x,np.ones(x.shape[0]))) 
    error=pi-y 
    grad_theta=np.dot(error.T,x_mod)  
   
    return grad_theta.flatten()

if __name__ == "__main__":
    # Ejemplo de uso
    np.random.seed(0)
    x = np.random.rand(10, 2)  # 100 muestras, 2 características
    print("x:",x)
    y = np.random.randint(0, 2, (10,1))  # Etiquetas binarias
    num_lo = 1  # Número de clases (1 para regresión logística binaria)
    var = np.random.rand(3)  # Parámetros iniciales (2 para theta, 1 para sesgo)    
    pi=fun_logistica(var,1,x)
    f=maxima_verosimilitud(pi,y)
    gra=gra_maxima_verosimilitud(pi,x,y)
    print("gra",gra)
    

