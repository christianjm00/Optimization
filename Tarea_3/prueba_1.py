class prueba:
    def __init__(self,a):
        self.a=a
        self.b=None
        self.i=0
        self.funcion=None
        self.var=None
    def fun_0(self,*var):
        self.b=self.i
        
    def fun_1(self, *var):
        self.var=var
        self.b=self.i**self.var[0]
    def suma(self,num):
        e=0
        
        for self.i in range(num):
            self.funcion()
            d=self.i*(self.a+self.b)
            e+=d
        return e
pru=prueba(3)
pru.funcion=pru.fun_0()
pru.suma(4)

pru=prueba(3)
pru.funcion=pru.fun_1(5)
pru.suma(4)