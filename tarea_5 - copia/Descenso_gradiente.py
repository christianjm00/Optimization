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

    def __init__(self,f,g,H,x_0,dat_X_train,dat_y_train,tam_lot,met_algor,ty_alpha,paso,error=10**(-5),num_i=100):
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
        self.ty_alpha=ty_alpha
        self.paso=paso
        self.fun_cal=None
        self.memor_error=[]
        self.memor_x=[]
        self.memor_f=[]
        self.memor_grad=[]
        self.fun_met_algo=[]
        if self.met_algor=="SIMPLE":
            if self.ty_alpha==1:
                self.fun_alpha=self.alpha_ty_1
            elif self.ty_alpha==2:
                self.fun_alpha=self.alpha_ty_2
            elif self.ty_alpha==3:      
                self.fun_alpha=self.alpha_ty_3
            self.fun_cal=self.simple
        if self.met_algor=="ADAM":
            self.fun_cal=self.ADAM
        elif self.met_algor=="ADAMW":
            self.fun_cal=self.ADAMW
        elif self.met_algor=="Nesterov":
            self.fun_cal=self.Nesterov
    
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

    
    def alpha_ty_2(self):
        """Método 2: Alpha basado en la aproximación de Newton-Raphson.""" 
        self.H_eva=self.H(self.memor_x[-1])
        den=np.dot(self.g_eva,np.dot(self.H_eva,self.g_eva))
        if den!=0:
            self.alpha=np.dot(self.g_eva,self.g_eva)/den
        else:
            self.alpha=self.paso

    
    def alpha_ty_3(self):
        """Método 3: Alpha basado en una regla de ajuste iterativa."""
        if self.i==0:
            self.alpha=self.paso
        else:
           
            den=2*(self.memor_f[-2]-self.f_i+self.alpha*np.dot(self.g_eva,self.g_eva))
            if den!=0:
                self.alpha=(self.alpha**2)*np.dot(self.g_eva,self.g_eva)/den
            else:
                self.alpha=self.paso

    def ADAM(self):
        if self.i==0:
            self.beta_1=0.9
            self.beta_2=0.95
            self.epsilon=10**(-8)
            m_tl1=np.zeros_like(self.g_eva)
            v_tl1=np.zeros_like(self.g_eva)
        else:
            m_tl1=self.m_t
            v_tl1=self.v_t
        self.m_t=self.beta_1*m_tl1+(1-self.beta_1)*self.g_eva 
        self.v_t=self.beta_2*v_tl1+(1-self.beta_2)*self.g_eva**2
        m_t_h=self.m_t/(1-self.beta_1**(self.i+1))
        v_t_h=self.v_t/(1-self.beta_2**(self.i+1))
        self.cal_cam=-(self.paso*m_t_h)/(np.sqrt(v_t_h)+self.epsilon)

    def ADAMW(self):
        
        if self.i==0:
            self.beta_1=0.9
            self.beta_2=0.95
            self.epsilon=10**(-8)
            self.lamb=10**(-3)
            m_tl1=np.zeros_like(self.g_eva)
            v_tl1=np.zeros_like(self.g_eva)
        else:
            m_tl1=self.m_t
            v_tl1=self.v_t
        self.m_t=self.beta_1*m_tl1+(1-self.beta_1)*self.g_eva 
        self.v_t=self.beta_2*v_tl1+(1-self.beta_2)*self.g_eva**2
        m_t_h=self.m_t/(1-self.beta_1**(self.i+1))
        v_t_h=self.v_t/(1-self.beta_2**(self.i+1))
        self.cal_cam=-self.paso*(m_t_h/(np.sqrt(v_t_h)+self.epsilon)+self.lamb*self.x_i)

    def Nesterov(self):
        if self.i==0:
            eta=0.1
            self.beta=0.5
            v_i=np.zeros_like(self.x_0)
        else:
            v_i=self.v_ip1
        
        y=self.x_i+self.beta*v_i
        caller=inspect.stack()[1].function  
        if caller == "run":
            g_to=self.g(y)
        elif caller == "run_es":
            g_to=self.g(y,self.dat_x_batch,self.dat_y_batch)

        self.v_ip1=self.beta*v_i-eta*g_to
        self.cal_cam=self.v_ip1

    def backtracking(self):
        """
        Método de backtracking para encontrar un paso alpha que satisfaga las condiciones de Wolfe.
        """
        if self.i==0:
            self.alpha=self.paso  # Valor inicial de alpha
        else:
            self.alpha=self.paso  # Reiniciar alpha en cada iteración
            self.beta=10**6  # Límite superior para alpha
            self.c_1=0.5  # Constante para la condición de Armijo
            self.c_2=0.9  # Constante para la condición de curvatura

            # Bucle de backtracking
            while True:
                # Calcular el nuevo punto
                x_ip1=self.x_i-self.alpha*self.g_eva
                
                # Evaluar la función en el nuevo punto
                caller=inspect.stack()[1].function
                if caller=="run":
                    f_ip1=self.f(x_ip1)
                    g_ip1=self.g(x_ip1)
                elif caller=="run_es":
                    f_ip1=self.f(x_ip1,self.dat_x_batch)
                    g_ip1=self.g(x_ip1,self.dat_x_batch,self.dat_y_batch)

                # Condición de Armijo
                armijo=f_ip1>=self.f_i-self.c_1*self.alpha*np.dot(self.g_eva,self.g_eva)

                # Condición de curvatura
                curvatura=np.dot(g_ip1,-self.g_eva)<=self.c_2*np.dot(self.g_eva,-self.g_eva)

                if armijo and curvatura:
                    break  # Alpha satisface ambas condiciones
                elif armijo:
                    self.beta=self.alpha  # Reducir el límite superior
                    self.alpha=(self.alpha+self.beta)/2
                elif curvatura:
                    self.alpha=(self.alpha+self.beta)/2  # Ajustar alpha

        self.cal_cam=-self.alpha*self.g_eva

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
if __name__ == "__main__":
    np.random.seed(42)
    print("experimento 0")
    def fun(x):
        return np.array(x**2 + 5*x + 7)
    
    def gra(x):
        return np.array(2*x + 5)
    
    def Hes(x):
        return np.array([[2]])
    for j in range(1,4):
        print("metodo num:",j)
        for i in range(5):
            punt_in=np.random.randint(-10,10)
            prueba=Descenso_gradiente(f=fun,g=gra,H=Hes,x_0=np.array([punt_in]),met_algor="SIMPLE",ty_alpha=j,paso=0.4,num_i=1000,error=10**(-12))
            resultado=prueba.run()
            print("punt_in:",punt_in)
            print("x_n:",resultado[0])
            print("error:",prueba.error)
            print("f(x_n):",resultado[1])
            print("i:",prueba.i)

        

    print("experimento 1")
    def fun(x):
        return np.array(x[0]**2+7*x[0]+x[1]**2-4*x[1]+3)
    def gra(x):
        return np.array([2*x[0]+7,2*x[1]-4])
    def Hes(x):
        return np.array([[2,0],[0,2]])
    for j in range(1,4):
        print("metodo num:",j)
        for i in range(5):
            punt_in=np.random.randint(-10,10,2)
            prueba_1=Descenso_gradiente(f=fun,g=gra,H=Hes,x_0=punt_in,met_algor="SIMPLE",ty_alpha=j,paso=0.4,num_i=1000,error=10**(-12))
            resultado_1=prueba_1.run()
            print("punt_in:",punt_in)
            print("x_n:",resultado_1[0])
            print("f(x_n):",resultado_1[1])
            print("i:",prueba_1.i)
            print("error:",prueba_1.error)

