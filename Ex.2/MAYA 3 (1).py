# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:19:25 2025

@author: Lenovo-pc
"""
import numpy as np
import matplotlib.pyplot as plt
import time

#%% Q1 ###

#%% a:
x = np.linspace(0,1,1000)
y = np.sin((100*x)**0.5) **2

plt.plot(x,y)
plt.title(r'$\sin^2$($\sqrt{100x}$)')

#%% b:    
#trapezoid rule: int = dx*(f_(i+1)-f_(i))/2  
#Newman 5.3: I_i1 = I_i/2 + h_i1*sum_odd(y_i1)    
N=1
x = np.linspace(0,1,N+1)
y = np.sin((100*x)**0.5) **2
A_vec = (x[1:]-x[:-1])*(y[1:]+y[:-1])/2
I_i1 = np.sum(A_vec)
e_i = I_i1/3
count = 0
while abs(e_i) > 1e-6:
    count +=1
    I_i = I_i1
    N = N*2
    
    x = np.linspace(0,1,N+1)
    y = np.sin((100*x)**0.5) **2
    I_i1 = I_i/2 + 1/N * np.sum(y[1::2])
    e_i = (I_i1-I_i)/3
    
print('#intervals', N, 'e_i', e_i, 'I', I_i1)
print('#iterations', count)
I_accurate = 0.455832532309
print('diff', I_accurate-I_i1)

#%% c:
def f(x):
    return np.sin((100*x)**0.5) **2

N=1
x = np.linspace(0,1,N+1)
A_vec = (x[1:]-x[:-1])*(f(x[1:])+f(x[:-1]))/2
I_i1 = np.sum(A_vec)
R11 = I_i1

N=2
x = np.linspace(0,1,N+1)
I_i2 = I_i1/2 + 1/N * np.sum(f(x[1::2]))
R21 = I_i2
R22 = R21 + (R21-R11)/3

e = 1
R_im = np.array([R21, R22])
# print(R11, R21, R22)
while abs(e)>1e-6:
    I_i1 = R_im[0]
    N = N*2
    x = np.linspace(0,1,N+1)
    I_i2 = I_i1/2 + 1/N * np.sum(f(x[1::2]))    
    
    e = 0
    R_im1 = np.zeros(len(R_im)+1)
    R_im1[0] = I_i2
    for n in range(len(R_im)):
        R_im1[n+1] = R_im1[n]+(R_im1[n]-R_im[n])/(4**(n+1)-1)
        # e += (R_im1[n]-R_im[n])/(4**(n+1)-1)
    e = (R_im1[-1]-R_im[-1])/(4**(len(R_im)+1)-1)
    R_im = R_im1
print('#intervals', N, 'e_i', e, 'I', R_im[-1])
print('#iterartions = ', len(R_im))
I_accurate = 0.455832532309
print('diff', abs(I_accurate-R_im[-1]))

#%% Q2

#%% a:
x = np.linspace(0.5,10,100)
f = np.sin(x-x**2)/x
noise = np.random.uniform(low=-1, high=1, size=(100))
a = 0.2
f_noisy = f+a*noise

plt.plot(x,f)
plt.plot(x,f_noisy)
plt.title(r'$\sin^2$($\sqrt{100x}$) + noise')

#%% b:
x = np.linspace(0.5,10,100)
f = np.sin(x-x**2)/x
f_derv = (f[2:]-f[:-2])/(x[2:]-x[:-2])  #central 1st order derivative
noise = np.random.uniform(low=-1, high=1, size=(100))
a_list = [0.01, 0.05, 0.1, 0.2]
# plt.plot(x[:-1],f_derv, label='no noise')

for a in a_list:
    f_noisy = f+a*noise
    f_noisy_derv = (f_noisy[2:]-f_noisy[:-2])/(x[2:]-x[:-2])
    plt.plot(x[1:-1],f_derv-f_noisy_derv, label=a)

plt.title(r'$f^{\prime}$(x)-$f^{\prime}_{noisy}$(x))')
plt.legend()

# as the data becomes noisier so does it derivative 

#%% c: - *to add comparison with b*
# p_i = c_i0 + c_i1*x + c_i2*x**2 + c_i3*x**3
N = 100
x = np.linspace(0.5,10,N)
f = np.sin(x-x**2)/x
noise = np.random.uniform(low=-1, high=1, size=(N))
const = 0.2
f_noisy = f+const*noise
f = f_noisy

# based on Slide 34:
dx = x[2]-x[1]
A = (np.eye(N-2, k=1) + np.eye(N-2, k=-1) + 4*np.eye(N-2, k=0))*dx
B = (f[:-2] -2*f[1:-1] +f[2:]) * 6/dx
p_2derv = np.linalg.solve(A, B)
p_2derv = np.append(np.array(0), p_2derv); p_2derv = np.append(p_2derv, np.array(0)) 

# based on Slide 31:
# p_i = a(x-x_i)^3 + b(x-x_(i+1))^3 + c*(x-x_i) + d*(x-x_(i+1))
a = p_2derv[1:]/6/dx
b = -p_2derv[:-1]/6/dx
c = -(p_2derv[1:]*dx**2 - 6*f[1:])/6/dx
d = (p_2derv[:-1]*dx**2 - 6*f[:-1])/6/dx

# p_i' = 3a(x-x_i)^2 + 3b(x-x_(i+1))^2 + c + d
x_vec = np.array([])
p_1derv_vec = np.array([])
for i in range(N-1):
    xi = np.linspace(x[i],x[i+1],10)
    p_1derv = 3*a[i]*(xi-x[i])**2 + 3*b[i]*(xi-x[i+1])**2 + c[i] + d[i]
    
    x_vec = np.append(x_vec, xi)
    p_1derv_vec = np.append(p_1derv_vec, p_1derv)
plt.plot(x_vec, p_1derv_vec)
plt.title(r'$p^{\prime}_{noisy}$(x))')
plt.show()

# f_derv-p1_derv (for comparison to b):
f = np.sin(x_vec-x_vec**2)/x_vec
f_derv = (f[2:]-f[:-2])/(x_vec[2:]-x_vec[:-2])  #central 1st order derivative
plt.plot(x_vec[1:-1],f_derv-p_1derv_vec[1:-1])
plt.title(r'$f^{\prime}$(x)-$p^{\prime}_{noisy}$(x))')

# # CHECK - original function vs spline:
# plt.show()
# for i in range(N-1):
#     xi = np.linspace(x[i],x[i+1],10)
#     spline = a[i]*(xi-x[i])**3 + b[i]*(xi-x[i+1])**3 + c[i]*(xi-x[i]) + d[i]*(xi-x[i+1])
#     plt.plot(xi, spline)
# x = np.linspace(0.5,10,100)
# plt.plot(x, np.sin(x-x**2)/x)
# plt.title('f_noisy - cubic spline - 0th derivative')

#%% Q3
# f = np.sin(x-x**2)/x
def F(x):
    return np.sin(x-x**2)/x
#%% Bisection:
# def F(x):
#     return np.sin(x-x**2)/x

intervals = [[0.5,10]]
zeros_found = []
list = []
d_min = 1e-3
start_time = time.perf_counter()
for x in intervals:
    x_l, x_u = x
    d = x_u - x_l
    while d > d_min:
        x_r = (x_l+x_u)/2
        mult = F(x_l)*F(x_r)
        d = x_r - x_l
        
        if d<d_min and F(x_l)*F(x_u)<=0:
            zeros_found.append(x_r)
            d = 0
            
        else:
            if mult<0:
                interval = [x_r,x_u]
                x_u = x_r
            elif mult>=0:
                interval = [x_l,x_r]
                x_l = x_r
            intervals.append(interval)
end_time = time.perf_counter()
print('t',end_time - start_time)

x = np.linspace(intervals[0][0], intervals[0][1], 1000)
plt.plot(x,F(x))
for x0 in zeros_found:
    plt.axvline(x=x0, ymin=0, ymax=1)
plt.axhline(y=0, xmin=0, xmax=1)
plt.show()

# #scatter plot of values of the 'zeros':
# for x0 in zeros_found:
#     plt.plot(x0,F(x0),'o')


#%% Newton-Raphson:
def F_div(x):
    return (np.cos(x-x**2)*(1-2*x)*x - np.sin(x-x**2)) / x**2

inter = [0.5,10]
N_intervals = 100
x = np.linspace(inter[0], inter[1],N_intervals)
N_max = 50
e = 1e-3
zeros_found = []
start_time = time.perf_counter()
for x0 in x:
    x1 = x0 - F(x0)/F_div(x0)
    N = 0
    while abs(x1-x0)>e*x0:
        N += 1
        x0 = x1
        x1 = x0 - F(x0)/F_div(x0)
        if N > N_max:
            break
        if F_div(x1) == 0:
            break
    if abs(x1-x0)<=e*x0 and x[0]<x1<x[-1]:
        zeros_found.append(x1)
    
#erase most repetitions:
end_time = time.perf_counter()
print('t',end_time - start_time)

zeros_found = np.round(zeros_found, decimals=3)
zeros_found = set(zeros_found)

x = np.linspace(inter[0], inter[1], 1000)
plt.plot(x,F(x))
for x0 in zeros_found:
    # if x0>0.5 and x0<10:
    plt.axvline(x=x0, ymin=0, ymax=1)
plt.axhline(y=0, xmin=0, xmax=1)

# plt.show()
# #scatter plot of values of the 'zeros':
# for x0 in zeros_found:
#     plt.plot(x0,F(x0),'o')

#%% Secant:

inter = [0.5,10]
N_intervals = 100
x = np.linspace(inter[0], inter[1],N_intervals)
# x0 = 2
N_max = 50
e = 1e-3
zeros_found = []

start_time = time.perf_counter()
for i in range(len(x)-1):
    x0 = x[i]; x1 = x[i+1]
    x2 = x1 - F(x1)*(x0-x1)/(F(x0)-F(x1))
    N = 0
    while abs(x1-x0)>e*x0:
        N += 1
        x0 = x1; x1 = x2
        x2 = x1 - F(x1)*(x0-x1)/(F(x0)-F(x1))
        if N > N_max:
            break
        if F_div(x1) == 0:
            break
    if abs(x1-x0)<=e*x0 and x[0]<x1<x[-1]:
        zeros_found.append(x1)
    
end_time = time.perf_counter()
print('t',end_time - start_time)


#erase most repetitions:
zeros_found = np.round(zeros_found, decimals=3)
zeros_found = set(zeros_found)

x = np.linspace(inter[0], inter[1], 1000)
plt.plot(x,F(x))
for x0 in zeros_found:
    if x0>0.5 and x0<10:
        # plt.plot(x0,F(x0),'o',color='r')
        plt.axvline(x=x0, ymin=0, ymax=1)
plt.axhline(y=0, xmin=0, xmax=1)

# plt.show()
# #scatter plot of values of the 'zeros':
# for x0 in zeros_found:
#     plt.plot(x0,F(x0),'o')








