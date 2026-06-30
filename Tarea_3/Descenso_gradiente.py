import numpy as np
import pickle
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

    def __init__(self,f,g,H,x_0,met_algor,ty_alpha,paso,error=10**(-5),num_i=100):
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
        """
        elif self.met_algor=="SGD_SIMPLE":
            print()
        elif self.met_algor=="SGD_Nesterov":    
            print()
        elif self.met_algor=="SGD_ADAM":
            print()
        elif self.met_algor=="SGD_ADAMW":
            print()  
        """
          
    
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
           
            den=2*(self.memor_f[-2]-self.memor_f[-1]+self.alpha*np.dot(self.g_eva,self.g_eva))
            if den!=0:
                self.alpha=(self.alpha**2)*np.dot(self.g_eva,self.g_eva)/den
            else:
                self.alpha=self.paso

    def ADAM(self):
        self.beta_1=0.9
        self.beta_2=0.95
        self.epsilon=10**(-8)
        if self.i==0:
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
        self.beta_1=0.9
        self.beta_2=0.95
        self.epsilon=10**(-8)
        self.lamb=10**(-3)
        if self.i==0:
            m_tl1=np.zeros_like(self.g_eva)
            v_tl1=np.zeros_like(self.g_eva)
        else:
            m_tl1=self.m_t
            v_tl1=self.v_t
        self.m_t=self.beta_1*m_tl1+(1-self.beta_1)*self.g_eva 
        self.v_t=self.beta_2*v_tl1+(1-self.beta_2)*self.g_eva**2
        m_t_h=self.m_t/(1-self.beta_1**(self.i+1))
        v_t_h=self.v_t/(1-self.beta_2**(self.i+1))
        self.cal_cam=-self.paso*(m_t_h/(np.sqrt(v_t_h)+self.epsilon)+self.lamb*self.memor_x[-1])

    def Nesterov(self):
        if self.i==0:
            v_i=np.zeros_like(self.x_0)
        else:
            v_i=self.v_ip1
        eta=0.1
        self.beta=0.5
        y=self.memor_x[-1]+self.beta*v_i
        self.v_ip1=self.beta*v_i-eta*self.g(y)
        self.cal_cam=self.v_ip1

    def run(self):
        """
        Ejecuta el algoritmo de descenso del gradiente.

        Devuelve:
        - Último punto encontrado (x_n), minimizador de f.
        - mínimo de f.
        """
        x_i=self.x_0
        f_i=self.f(x_i)
        self.memor_x.append(x_i)
        self.memor_f.append(f_i)
        for self.i in range(self.num_i):
            self.g_eva=self.g(x_i)
            self.fun_cal()
            x_ip1=x_i+self.cal_cam
            f_ip1=self.f(x_ip1)
            error_i=np.linalg.norm(self.g_eva)
            self.memor_error.append(error_i)
            self.memor_x.append(x_ip1)
            self.memor_f.append(f_ip1)
            self.memor_grad.append(self.g_eva)
            if error_i<=self.error:
                break
            
            x_i=x_ip1
            f_i=f_ip1
        self.x_n=x_ip1  
        return x_ip1,f_i

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

