import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
from IPython.display import HTML

# ======================
# PARÁMETROS FÍSICOS
# ======================
L = 3.0            # Longitud (m)
h0 = 0.15          # Altura en x=0 (m)
E = 70e9           # Módulo de Young (Aluminio)
rho = 2700         # Densidad (kg/m³)
c = 20             # Amortiguamiento (N·s/m²)
F0 = 800           # Amplitud de fuerza (N)
freq = 4           # Frecuencia de excitación (Hz)
ne = 20            # Número de elementos
N = 5              # Orden polinomial por elemento

# ======================
# MALLA ESPECTRAL
# ======================
# Puntos de Gauss-Lobatto-Legendre por elemento
xi, weights = np.polynomial.legendre.leggauss(N)
xi = np.sort(xi)  # Puntos en [-1, 1]

# Matrices de elemento
def elemental_matrices(xe):
    h = xe[-1] - xe[0]
    J = h/2  # Jacobiano
    
    # Derivadas de las funciones de base
    dphi = np.zeros((N, N))
    phi = np.zeros((N, N))
    for i in range(N):
        coeff = np.zeros(N)
        coeff[i] = 1
        phi[i] = np.polynomial.legendre.legval(xi, coeff)
        dphi[i] = np.polynomial.legendre.legval(xi, np.polynomial.legendre.legder(coeff))
    
    # Matrices locales
    Me = np.zeros((N, N))
    Ke = np.zeros((N, N))
    Ce = np.zeros((N, N))
    
    for k in range(N):
        for l in range(N):
            Me[k,l] = weights[k] * phi[k] * phi[l] * J
            Ke[k,l] = weights[k] * dphi[k] * dphi[l] / J
            Ce[k,l] = weights[k] * phi[k] * phi[l] * J
    
    return Me, Ke, Ce

# ======================
# ENSAMBLAJE GLOBAL
# ======================
def assemble_system():
    # Coordenadas globales
    nodes = np.linspace(0, L, ne+1)
    elements = [nodes[i:i+2] for i in range(ne)]
    
    # Dimensiones
    ndof = ne*(N-1) + 1
    M = lil_matrix((ndof, ndof))
    K = lil_matrix((ndof, ndof))
    C = lil_matrix((ndof, ndof))
    
    # Propiedades materiales variables
    def EI(x):
        h = h0*(1 - 0.3*x/L)
        return E * (h**4)/12
    
    def rhoA(x):
        h = h0*(1 - 0.3*x/L)
        return rho * (h**2)
    
    # Ensamblaje
    for e, xe in enumerate(elements):
        Me, Ke, Ce = elemental_matrices(xe)
        x_local = 0.5*(xe[0] + xe[1]) + 0.5*(xe[1]-xe[0])*xi
        
        # Factores variables
        EI_vals = EI(x_local)
        rhoA_vals = rhoA(x_local)
        
        # Ensamblar con factores
        Me_scaled = np.diag(rhoA_vals) @ Me
        Ke_scaled = np.diag(EI_vals) @ Ke
        Ce_scaled = c * Ce
        
        # Mapeo de índices
        global_dofs = [e*(N-1) + i for i in range(N)]
        
        for i, gi in enumerate(global_dofs):
            for j, gj in enumerate(global_dofs):
                M[gi, gj] += Me_scaled[i,j]
                K[gi, gj] += Ke_scaled[i,j]
                C[gi, gj] += Ce_scaled[i,j]
    
    return M.tocsc(), K.tocsc(), C.tocsc(), nodes

# ======================
# SIMULACIÓN TEMPORAL
# ======================
def run_simulation():
    M, K, C, nodes = assemble_system()
    ndof = M.shape[0]
    
    # Condiciones iniciales
    u0 = np.zeros(ndof)
    v0 = np.zeros(ndof)
    
    # Parámetros temporales
    dt = 1e-4
    T = 1.0
    steps = int(T/dt)
    t = np.linspace(0, T, steps)
    
    # Almacenamiento
    U = np.zeros((steps, ndof))
    
    # Matriz del sistema (Newmark-beta)
    beta = 0.25
    gamma = 0.5
    A = M + gamma*dt*C + beta*dt**2*K
    A = A.tocsc()
    
    # Fuerza externa
    def F(t, nodes):
        force = np.zeros_like(nodes)
        force[-1] = F0 * np.sin(2*np.pi*freq*t)  # Fuerza en extremo
        return force
    
    # Integración temporal
    print("⏳ Ejecutando simulación...")
    for i in range(1, steps):
        # Predictor
        u_pred = U[i-1] + dt*v0 + (0.5-beta)*dt**2*(M@v0)
        
        # Vector de carga
        f_ext = F(t[i], nodes)
        b = f_ext - C@v0 - K@u_pred
        
        # Resolver
        du = spsolve(A, b)
        
        # Corrector
        U[i] = u_pred + beta*dt**2*du
        v0 = v0 + gamma*dt*du
        
        if i % 100 == 0:
            print(f"Progreso: {100*i/steps:.1f}%")
    
    return U, nodes, t

# ======================
# VISUALIZACIÓN
# ======================
def animate_results(U, nodes, t):
    fig, ax = plt.subplots(figsize=(12,6))
    line, = ax.plot(nodes, U[0,:len(nodes)], 'r-', linewidth=2)
    
    ax.set_xlim(0, L)
    ax.set_ylim(-1.2e-4, 1.2e-4)
    ax.set_xlabel('Posición (m)', fontsize=12)
    ax.set_ylabel('Desplazamiento (m)', fontsize=12)
    ax.grid(True)
    ax.set_title('Simulación Espectral de Viga Cónica', fontsize=14)
    
    def update(frame):
        line.set_ydata(U[frame,:len(nodes)])
        ax.set_title(f'Viga cónica - t = {t[frame]:.3f} s', fontsize=14)
        return line,
    
    ani = animation.FuncAnimation(
        fig, update, frames=range(0, len(t), 10), 
        blit=True, interval=50
    )
    
    plt.close()
    return ani

# ======================
# EJECUCIÓN
# ======================
U, nodes, t = run_simulation()
print("🎉 Simulación completada!")
ani = animate_results(U, nodes, t)
HTML(ani.to_html5_video())