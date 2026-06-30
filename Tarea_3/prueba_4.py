class prueba:
    def __init__(self, a):
        self.a = a
        self.b = None
        self.i = 0
        self.funcion = None
        self.var = None

    def fun_0(self, *var):
        self.b = self.i

    def fun_1(self, *var):
        self.var = var
        print(self.var)  # Para ver qué se pasa a la función
        self.b = self.i ** self.var[0]

    def suma(self, num):
        e = 0
        for self.i in range(num):
            print(self.i)  # Ver los valores de i en cada iteración
            self.funcion()  # Aquí se llama a la función asignada
            d = self.i * (self.a + self.b)
            e += d
        return e

# Crear objeto
pru = prueba(3)

# Asignar la función a usar, sin llamar la función
pru.funcion = pru.fun_0

# Llamar al método suma que usará la función asignada
resultado_0 = pru.suma(4)
print("Resultado con fun_0:", resultado_0)

# Crear nuevo objeto
pru_1 = prueba(3)

# Asignar la función, pasando el parámetro que necesita
pru_1.funcion = lambda: pru_1.fun_1(2)  # Asignar una función que llame a `fun_1` con el parámetro 2

# Llamar al método suma que usará la nueva función asignada
resultado_1 = pru_1.suma(4)
print("Resultado con fun_1:", resultado_1)