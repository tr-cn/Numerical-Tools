####### HW 4 ########

import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng() 


#%% Q1 - 2D poisson
plt.close()

#let's choose a Face Centered Grid
def BC_phi(phi,BC):
    if BC == 0: #Diriclet
        phi[[0,-1],:] = 0     
        phi[:,[0,-1]] = 0    
    elif BC == 1:
        phi[0,:] = phi[-1,:]  
        phi[:,0] = phi[:,-1]
    
    return phi    

def relaxation (N, BC):
    Nx = N; Ny = N
    x = np.linspace(-1,1,Nx)
    y = np.linspace(-1,1,Ny)
    dx = x[1]-x[0]
    X,Y = np.meshgrid(x,y)
    phi_1 = np.zeros([Nx,Ny])
    f = np.zeros([Nx,Ny]); f[int(Nx/2), int(Ny/2)] = -1/dx**2
    
    # plt.plot()            
    # plt.contourf(x, y, f, label = 'f')
    # plt.xlabel('x', fontsize=18)
    # plt.ylabel('y', fontsize=18)
    # plt.title('Poisson - Dirichlet BC - f', fontsize=18)
    # plt.colorbar()
    # plt.show()
    
    eps = [1]
    while eps[-1]>e_stop:     
        phi_0 = phi_1.copy()
        phi_1 = phi_0 * 0  
        
        for i in range(1,len(phi_0[:,0])-1):
            for j in range(1,len(phi_0[0,:])-1):            
                phi_1[i,j] = (phi_0[i+1,j] + phi_0[i-1,j] + phi_0[i,j+1] + phi_0[i,j-1] - f[i,j] *dx**2) /4
        #BC
        phi_1 = BC_phi(phi_1,BC)
        
        # eps.append( np.sqrt(np.sum((phi_1-phi_0)**2) /np.size(phi_1) ) )
        eps.append( np.max(abs(phi_1-phi_0)) )
    
    if BC ==0:
        Title_phi = 'Diriclet BC (phi=0)'
    elif BC == 1:
        Title_phi = 'Periodic BC'
    
    plt.plot()   
    plt.contourf(x, y, phi_1, label = f"{Title_phi}, N={len(eps)-1}")
    plt.title(f"{Title_phi}, N={len(eps)-1}", fontsize=18)
    plt.colorbar()
    plt.show()
    
    plt.figure
    plt.plot(range(len(eps)-1), eps[1:])
    plt.title(f"error - {Title_phi}, N={len(eps)-1}", fontsize=18)
    plt.xscale('log') 
    plt.yscale('log')
    plt.show()
    
    print(f"N where e<{e_stop}: ", len(eps)-1)

e_stop = 1e-5
N = 34
relaxation (N, 0)   #Diriclet BC
relaxation (N, 1)   #Periodic BC


#%% Q2  - Upwind -  WRONG CALC OF v_shock?: I've taken x(t_i)/t_i ("ave") instead of v(t_i) ("instantaneous")
#a: du/dt + u*du/dx = 0
#b: du/dt + 0.5*du^2/dx = 0

def step(x, a_vec, method, dx,dt):
    a_vec_new = a_vec*0
    if method == 1:     #du/dt + u*du/dx = 0
        for i in range(len(x)):
            a_vec_new[i+1] = a_vec[i+1] - (a_vec[i+1]-a_vec[i]) *a_vec[i+1]*dt/dx
    elif method == 2:   #du/dt + 0.5*du^2/dx = 0        
        for i in range(len(x)):
            a_vec_new[i+1] = a_vec[i+1] - (a_vec[i+1]**2-a_vec[i]**2) *0.5 *dt/dx 
    return a_vec_new
        
def upwind (method):
    CFL = 0.5  #= u*dt/dx
    # h = x[1]-x[0]
    # u = 0.1
    # dt = CFL *h/u
    # t_end = 1/u 
    
    t_list = [20,40,60] #,80]  #t_print
    
    #initial wave
    N = 64
    x = np.linspace(0,1,N)
    a_vec_0 = np.concatenate((2*np.ones(int(N/2)+1),np.ones(int(N/2)+1)))  
    
    CFL = 0.5  #= u*dt/dx
    dx = x[1]-x[0]
    dt = CFL*dx/np.max(a_vec_0)
    N_iterations = t_list[-1]  
    
    #measure shock velocity
    res_list = [0.99, 0.5, 0.01]
    inx_res1 = []; inx_res2 = []; inx_res3 = []
    inx_res_list = [inx_res1, inx_res2, inx_res3]
    
    a_vec = a_vec_0
    
    plt.figure()
    plt.title('advection - Upwind')
    plt.plot(x,a_vec_0[1:-1], label = 'initial')
    
    for n in range(N_iterations+1): 
        a_vec[0] = a_vec[2] #outflowing
        a_vec[1] = a_vec[2]
        a_vec[-1] = a_vec[-3]
        a_vec[-2] = a_vec[-3]
        
        a_vec_new = step(x, a_vec, method, dx,dt)
        a_vec = a_vec_new.copy()
        
        #velocity calc
        for i_res in range(len(res_list)):
            res = 1+res_list[i_res] 
            inx_res_i = inx_res_list[i_res]
            inx = np.where(a_vec<res)[0][1]
            inx_res_i.append(inx)
    
        if n in t_list:       
            plt.plot(x,a_vec_new[1:-1], label = f"N={n}")
            
            
    plt.legend()    
    plt.show()
    
    v_shock_list = []
    for i_list in range(len(inx_res_list)):
        inx_res_i = inx_res_list[i_list]
        d_inx_shock = np.array(inx_res_i[1:]) - inx_res_i[0]
        t = np.arange(len(d_inx_shock))+1
        v_shock = d_inx_shock/t
        
        v_shock_list.append(v_shock)
        plt.plot(range(N_iterations),v_shock, label=f"res={res_list[i_list]}")
    plt.legend()  
    plt.show()
    print(f"method {method} - v_shock =", v_shock_list[-1][-1])

upwind(method=1)   #a: du/dt + u*du/dx = 0
upwind(method=2)   #b: du/dt + 0.5*du^2/dx = 0


#%% Q3

def boris_step(v, E_field, B_field, dt):  
    T = B_field * 0.5 * dt;
    S = 2. * T / (1. + T*T);
    v_minus = v + E_field * dt/2;
    v_prime = v_minus + np.cross(v_minus,T);
    v_plus = v_minus + np.cross(v_prime,S);
    v_new = v_plus + E_field * dt/2;

    return v_new

def charge_motion(Ey):
    #dr/dt = v
    #dv/dt = (E + v \cross B) - Boris
    
    Bz = 1    
    B_field = np.array([0,0,Bz])
    E_field = np.array([0,Ey,0])
    
    path_x = []
    path_y = []
    
    dt = 0.1; t_end = 100
    
    r0 = np.array([0,1,0]);   v0 = np.array([1,0,0])
    v05 = v0 + E_field * dt/2
    v05 = v05 + np.cross(v05, B_field) *dt/2   #v(x,y) \cross Bz
    
    for i in range(int(t_end/dt)):
        r1 = r0 + v05*dt
        v15 = boris_step(v05, E_field, B_field, dt)
        
        path_x.append(r1[0])
        path_y.append(r1[1])

        r0 = r1.copy()
        v05 = v15.copy()
        
    #frequency and radius
    y_vec = np.array(path_y) 
    dy_vec = y_vec[1:]-y_vec[:-1]
    sgn0 = np.sign(dy_vec[0])
    inx_ext_list = []
    for i in range(len(dy_vec)):
        if np.sign(dy_vec[i])== -sgn0:
            inx_ext_list.append(i)   #min or max
            sgn0 = -sgn0
        if len(inx_ext_list)>4:
            break
    dt_round = dt*(inx_ext_list[2]-inx_ext_list[0])
        
    if Ey == 0:
        radius = abs(y_vec[inx_ext_list[1]+1]-y_vec[inx_ext_list[0]+1])/2
        print(f"E=0:  radius = {radius} cm")
        
        freq = 1/dt_round
        print(f"E=0:  frequency = {freq} Hz")
    else:
        dx_round = path_x[inx_ext_list[2]+1]-path_x[inx_ext_list[0]+1]
        v_drift = dx_round/dt_round        
        print(f"Ey=-0.5:  drift_velocity = {v_drift} cm/s")
    
    plt.plot(path_x,path_y)
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.title(f"Ey = {Ey}")
    plt.show()


charge_motion(Ey = 0)     #a
charge_motion(Ey = -0.5)  #b
