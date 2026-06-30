import numpy as np
import scipy 
import problema_1 as pro_1
import matplotlib.pyplot as plt
import time as tm
def backtracking(fun,x,fun_xk,p,grad_xk,alpha,c_1,rho,inter_b):
    alpha_i=np.copy(alpha)
    for i in range(inter_b):
        fun_alphai=fun(x+alpha_i*p)
        cond=fun_alphai<=fun_xk+c_1*alpha_i*np.dot(grad_xk,p)
        if cond:
            return  alpha_i, i
        alpha_i*=rho
    return  max(alpha_i,1e-8), i


def metod_newton_mod(fun,grad,hess,x_0,inter,tole,alpha,c_1,rho,inter_b,delta,gu_mem=True):
    x=x_0
    m=0
    res=0
    v_x=[]
    v_g=[]
    v_f=[]
    epsilon_m=np.finfo(float).eps
    epsilon=np.sqrt(epsilon_m)
    for i in range(inter):
        if tole==None:
            tole=np.sqrt(epsilon*(i+1))
        g=np.copy(grad(x))
        norm_g=np.copy(np.linalg.norm(g))
        f=np.copy(fun(x))
        if gu_mem:
            v_x.append(np.copy(x))
            v_g.append(np.copy(g))
            v_f.append(np.copy(f))
        if norm_g<tole:
            res=1
            break
        else:
            H=hess(x)
            eig_1,eig_n=pro_1.cal_eigval(H)
            type_class=pro_1.class_eigval(eig_1,eig_n,False)
            
            cond_1=eig_1<epsilon
            if cond_1:
                mod=delta+np.abs(eig_1)
                H+=mod*np.eye(len(H))
            if type_class==0:
                L_u,L_d=scipy.linalg.cho_factor(H)
                p=scipy.linalg.cho_solve((L_u,L_d),-g)
            else:
                p=-g
                m+=1

        alpha,j=backtracking(fun,x,f,p,g,alpha,c_1,rho,inter_b)
        x+=alpha*p
    if gu_mem:
        return x,g,i,m,res,f,np.array(v_x),np.array(v_g),np.array(v_f)
    else:
        return x,g,i,m,res,f

if __name__ == "__main__":
    print("resultado beale delta=0.001")
    x_beale_0=np.array([2.0,3.0])
    x_0=np.linspace(-0.5,3.6,200)
    x_1=np.linspace(-0.5,3.6,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_beale(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_beale,g_beale,i_beale,m_beale,res_beale,f_beale,v_x_beale,v_g_beale,v_f_beale=metod_newton_mod(pro_1.fun_beale,pro_1.grad_fun_beale,pro_1.Hess_fun_beale,np.copy(x_beale_0),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_beale(x_beale_0))
    print("iterracciones:",i_beale)
    print("m:",m_beale)
    print("res:",res_beale)
    print("norma de g_k",np.linalg.norm(g_beale))
    if len(v_x_beale)<=6:
        print(v_x_beale)
    else:
        print(v_x_beale[:3])
        print(v_x_beale[-3:])
    print("f(x_k):",v_f_beale[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_beale[:,0],v_x_beale[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Beale $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_1_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_beale)),np.linalg.norm(v_g_beale,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Beale $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_2_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_beale[:,0],v_x_beale[:,1],v_f_beale,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Beale $\delta=0.001$')
    plt.savefig("Gra_3_ta_8_pr_3.pdf")

    print("resultado himmelbau delta=0.001")
    x_himme_0=np.array([2.0,4.0])
    x_0=np.linspace(1.5,4.5,200)
    x_1=np.linspace(1.5,4.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Himmelbau(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_himme,g_himme,i_himme,m_himme,res_himme,f_himme,v_x_himme,v_g_himme,v_f_himme=metod_newton_mod(pro_1.fun_Himmelbau,pro_1.grad_fun_Himmelbau,pro_1.Hess_fun_Himmelbau,np.copy(x_himme_0),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Himmelbau(x_himme_0))
    print("iterracciones:",i_himme)
    print("m:",m_himme)
    print("res:",res_himme)
    print("norma de g_k",np.linalg.norm(g_himme))
    if len(v_x_himme)<=6:
        print(v_x_himme)
    else:
        print(v_x_himme[:3])
        print(v_x_himme[-3:])
    print("f(x_k):",v_f_himme[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_himme[:,0],v_x_himme[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Himmelbau $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_4_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_himme)),np.linalg.norm(v_g_himme,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Himmelbau $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_5_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_himme[:,0],v_x_himme[:,1],v_f_himme,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Himmelbau $\delta=0.001$')
    plt.savefig("Gra_6_ta_8_pr_3.pdf")


    print("resultado Rosenbrock 2D delta=0.001")
    x_Rose_0=np.array([-1.2,1.0])
    x_0=np.linspace(-1.5,1.5,200)
    x_1=np.linspace(-1.5,1.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Rosenbrock(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rose_0),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rose_0))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_Rose[:,0],v_x_Rose[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Rosenbrock 2D $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_7_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 2D $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_8_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_Rose[:,0],v_x_Rose[:,1],v_f_Rose,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Rosenbrock 2D $\delta=0.001$')
    plt.savefig("Gra_9_ta_8_pr_3.pdf")


    print("resultado Rosenbrock 200D delta=0.001")
    x_Rosen_0_1=np.ones(200)
    x_Rosen_0_1[0]=-1.2
    x_Rosen_0_1[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_1),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_1))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 200D $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_10_ta_8_pr_3.pdf")

    print("resultado Rosenbrock 600D delta=0.001")
    x_Rosen_0_2=np.ones(600)
    x_Rosen_0_2[0]=-1.2
    x_Rosen_0_2[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_2),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_2))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 600D $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_11_ta_8_pr_3.pdf")

    print("resultado hartmann delta=0.001")
    x_hart_0=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    def f_hart(x):
        return pro_1.fun_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    def g_hart(x):
        return pro_1.grad_fun_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    def h_hart(x):
        return pro_1.hess_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    tm_0=tm.time()
    x_hart,ge_hart,i_hart,m_hart,res_hart,fu_hart,v_x_hart,v_g_hart,v_f_hart=metod_newton_mod(f_hart,g_hart,h_hart,np.copy(x_hart_0),10000,None,1.0,0.1,0.6,100,0.001)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",f_hart(x_hart_0))
    print("iterracciones:",i_hart)
    print("m:",m_hart)
    print("res:",res_hart)
    print("norma de g_k",np.linalg.norm(ge_hart))
    if len(v_x_hart)<=6:
        print(v_x_hart)
    else:
        print(v_x_hart[:3])
        print(v_x_hart[-3:])
    print("f(x_k):",v_f_hart[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_hart)),np.linalg.norm(v_g_hart,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Hartmann $\delta=0.001$")
    plt.legend()
    plt.savefig("Gra_12_ta_8_pr_3.pdf")



    print("delta=10.0")
    print("resultado beale delta=0.001")
    x_beale_0=np.array([2.0,3.0])
    x_0=np.linspace(-0.5,3.6,200)
    x_1=np.linspace(-0.5,3.6,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_beale(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_beale,g_beale,i_beale,m_beale,res_beale,f_beale,v_x_beale,v_g_beale,v_f_beale=metod_newton_mod(pro_1.fun_beale,pro_1.grad_fun_beale,pro_1.Hess_fun_beale,np.copy(x_beale_0),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_beale(x_beale_0))
    print("iterracciones:",i_beale)
    print("m:",m_beale)
    print("res:",res_beale)
    print("norma de g_k",np.linalg.norm(g_beale))
    if len(v_x_beale)<=6:
        print(v_x_beale)
    else:
        print(v_x_beale[:3])
        print(v_x_beale[-3:])
    print("f(x_k):",v_f_beale[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_beale[:,0],v_x_beale[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Beale $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_13_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_beale)),np.linalg.norm(v_g_beale,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Beale $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_14_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_beale[:,0],v_x_beale[:,1],v_f_beale,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Beale $\delta=10.0$')
    plt.savefig("Gra_15_ta_8_pr_3.pdf")

    print("resultado himmelbau delta=10.0")
    x_himme_0=np.array([2.0,4.0])
    x_0=np.linspace(1.5,4.5,200)
    x_1=np.linspace(1.5,4.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Himmelbau(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_himme,g_himme,i_himme,m_himme,res_himme,f_himme,v_x_himme,v_g_himme,v_f_himme=metod_newton_mod(pro_1.fun_Himmelbau,pro_1.grad_fun_Himmelbau,pro_1.Hess_fun_Himmelbau,np.copy(x_himme_0),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Himmelbau(x_himme_0))
    print("iterracciones:",i_himme)
    print("m:",m_himme)
    print("res:",res_himme)
    print("norma de g_k",np.linalg.norm(g_himme))
    if len(v_x_himme)<=6:
        print(v_x_himme)
    else:
        print(v_x_himme[:3])
        print(v_x_himme[-3:])
    print("f(x_k):",v_f_himme[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_himme[:,0],v_x_himme[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Himmelbau $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_16_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_himme)),np.linalg.norm(v_g_himme,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Himmelbau $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_17_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_himme[:,0],v_x_himme[:,1],v_f_himme,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Himmelbau $\delta=10.0$')
    plt.savefig("Gra_18_ta_8_pr_3.pdf")


    print("resultado Rosenbrock 2D delta=10.0")
    x_Rose_0=np.array([-1.2,1.0])
    x_0=np.linspace(-1.5,1.5,200)
    x_1=np.linspace(-1.5,1.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Rosenbrock(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rose_0),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rose_0))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_Rose[:,0],v_x_Rose[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Rosenbrock 2D $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_19_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 2D $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_20_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_Rose[:,0],v_x_Rose[:,1],v_f_Rose,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Rosenbrock 2D $\delta=10.0$')
    plt.savefig("Gra_21_ta_8_pr_3.pdf")


    print("resultado Rosenbrock 200D delta=10.0")
    x_Rosen_0_1=np.ones(200)
    x_Rosen_0_1[0]=-1.2
    x_Rosen_0_1[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_1),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_1))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 200D $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_22_ta_8_pr_3.pdf")

    print("resultado Rosenbrock 600D delta=10.0")
    x_Rosen_0_2=np.ones(600)
    x_Rosen_0_2[0]=-1.2
    x_Rosen_0_2[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_2),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_2))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 600D $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_23_ta_8_pr_3.pdf")

    print("resultado hartmann delta=10.0")
    x_hart_0=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    def f_hart(x):
        return pro_1.fun_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    def g_hart(x):
        return pro_1.grad_fun_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    def h_hart(x):
        return pro_1.hess_hartmann(x,pro_1.alpha,pro_1.A,pro_1.P)
    tm_0=tm.time()
    x_hart,ge_hart,i_hart,m_hart,res_hart,fu_hart,v_x_hart,v_g_hart,v_f_hart=metod_newton_mod(f_hart,g_hart,h_hart,np.copy(x_hart_0),10000,None,1.0,0.1,0.6,100,10.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",f_hart(x_hart_0))
    print("iterracciones:",i_hart)
    print("m:",m_hart)
    print("res:",res_hart)
    print("norma de g_k",np.linalg.norm(ge_hart))
    if len(v_x_hart)<=6:
        print(v_x_hart)
    else:
        print(v_x_hart[:3])
        print(v_x_hart[-3:])
    print("f(x_k):",v_f_hart[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_hart)),np.linalg.norm(v_g_hart,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Hartmann $\delta=10.0$")
    plt.legend()
    plt.savefig("Gra_24_ta_8_pr_3.pdf")


    print("resultado Rosenbrock 2D delta=100.0")
    x_Rose_0=np.array([-1.2,1.0])
    x_0=np.linspace(-1.5,1.5,200)
    x_1=np.linspace(-1.5,1.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Rosenbrock(np.array([x_0,x_1]))
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rose_0),10000,None,1.0,0.1,0.6,100,100.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rose_0))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(v_x_Rose[:,0],v_x_Rose[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(r"Gráfica de Rosenbrock 2D $\delta=100.0$")
    plt.legend()
    plt.savefig("Gra_25_ta_8_pr_3.pdf")

    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 2D $\delta=100.0$")
    plt.legend()
    plt.savefig("Gra_26_ta_8_pr_3.pdf")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(v_x_Rose[:,0],v_x_Rose[:,1],v_f_Rose,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title(r'Gráfica de Rosenbrock 2D $\delta=100.0$')
    plt.savefig("Gra_27_ta_8_pr_3.pdf")

    print("resultado Rosenbrock 200D delta=100.0")
    x_Rosen_0_1=np.ones(200)
    x_Rosen_0_1[0]=-1.2
    x_Rosen_0_1[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_1),10000,None,1.0,0.1,0.6,100,100.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_1))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 200D $\delta=100.0$")
    plt.legend()
    plt.savefig("Gra_28_ta_8_pr_3.pdf")

    print("resultado Rosenbrock 600D delta=100.0")
    x_Rosen_0_2=np.ones(600)
    x_Rosen_0_2[0]=-1.2
    x_Rosen_0_2[-2]=-1.2
    tm_0=tm.time()
    x_Rose,g_Rose,i_Rose,m_Rose,res_Rose,f_Rose,v_x_Rose,v_g_Rose,v_f_Rose=metod_newton_mod(pro_1.fun_Rosenbrock,pro_1.gra_fun_Rosenbrock,pro_1.Hess_fun_Rosenbrock,np.copy(x_Rosen_0_2),10000,None,1.0,0.1,0.6,100,100.0)
    tm_1=tm.time()
    print("time:",tm_1-tm_0)
    print("f(x_0):",pro_1.fun_Rosenbrock(x_Rosen_0_2))
    print("iterracciones:",i_Rose)
    print("m:",m_Rose)
    print("res:",res_Rose)
    print("norma de g_k",np.linalg.norm(g_Rose))
    if len(v_x_Rose)<=6:
        print(v_x_Rose)
    else:
        print(v_x_Rose[:3])
        print(v_x_Rose[-3:])
    print("f(x_k):",v_f_Rose[-1])
    plt.figure()
    plt.scatter(np.arange(len(v_g_Rose)),np.linalg.norm(v_g_Rose,axis=1),label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Rosenbrock 600D $\delta=100.0$")
    plt.legend()
    plt.savefig("Gra_29_ta_8_pr_3.pdf")
    """
    prue=Descenso_gradiente(f=pro_1.fun_Himmelbau,g=pro_1.grad_fun_Himmelbau,H=pro_1.Hess_fun_Himmelbau,x_0=x_himme_0,dat_X_train=None,dat_y_train=None,tam_lot=None,met_algor="ADAMW",ty_alpha=None,paso=0.5)
    prue.run()
    print("prue.x_n",prue.x_n)
    print("prue.error",prue.error)
    print("prue.f:",prue.memor_f[0])
    print("prue.f:",prue.memor_f[-1])
    x_prue=np.array(prue.memor_x)
    x_0=np.linspace(1.5,4.5,200)
    x_1=np.linspace(1.5,4.5,200)
    x_0,x_1=np.meshgrid(x_0,x_1)
    f=pro_1.fun_Himmelbau(np.array([x_0,x_1]))
    plt.figure()
    contours=plt.contour(x_0, x_1, f, levels=7, colors='black')  # Líneas de contorno
    plt.clabel(contours, inline=True, fontsize=8) 
    plt.plot(x_prue[:,0],x_prue[:,1],label="x")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title("Gráfica de Hime")
    plt.legend()


    plt.figure()
    plt.scatter(np.arange(len(prue.memor_grad)),prue.memor_error,label="error")
    plt.xlabel("iteracciones")
    plt.ylabel(r"$\|\nabla f(x_k)\|$")
    plt.title(r"Gráfica $\|\nabla f(x_k)\|$ de Beale")
    plt.legend()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_0, x_1, f, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.plot(x_prue[:,0],x_prue[:,1],prue.memor_f,linewidth=5,label="punto evaluacion")
    # Etiquetas
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'f')
    ax.set_title('Gráfica de Himmelbau')
   

    #plt.show()
    """
    