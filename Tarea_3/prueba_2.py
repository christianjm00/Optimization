class prueba:
    def __init__(self, a, tipo_funcion='fun_0'):
        self.a = a
        self.b = None
        self.i = 0
        
        # Al crear el objeto, se asigna directamente el método adecuado
        self.funcion = getattr(self, tipo_funcion)  # Seleccionamos dinámicamente el método

    def fun_0(self):
        self.b = self.i

    def fun_1(self, b):
        self.b = self.i ** b

    def suma(self, num):
        e = 0
        for self.i in range(num):
            d = self.i * (self.a + self.b)
            e += d
        return e

    def aplicar_funcion(self, *args, **kwargs):
        # Aplicamos la función elegida
        self.funcion(*args, **kwargs)

# Ejemplo de uso:

# Usando 'fun_0'
pru_0 = prueba(3, tipo_funcion='fun_0')
pru_0.aplicar_funcion()  # Llama a fun_0 sin parámetros
print(pru_0.suma(4))  # Resultado con b=0

# Usando 'fun_1'
pru_1 = prueba(3, tipo_funcion='fun_1')
pru_1.aplicar_funcion(5)  # Llama a fun_1 con el parámetro 5
print(pru_1.suma(4))  # Resultado con b=0