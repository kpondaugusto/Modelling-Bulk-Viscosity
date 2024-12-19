#%%
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton, brentq
import matplotlib.pyplot as plt
#%%
### constants ###
eta = 1e-2
K = 1
gamma = 5/3
mdot = 1
mu = 0.01

### inital values ###
x0 = 0.1
xf = 3

v0 = 1
a0 = 0

rtol = 1e-6
atol = 1e-6

initial_conditions_v = [v0]
initial_conditions = [v0, a0]

x_values = np.linspace(x0,xf,1024)

### define the area ### 
def A(x):
    return x +1
def dA(x):
    return np.ones_like(x)

### find system of equations ###
def system_of_equations_novisc(x,v):
        # Adding a safeguard to avoid divisions by very small numbers
    if A(x) < 1e-6 or v < 1e-6:
        return 0  # return 0 if A(x) or v is too small (prevent instability)

    mdot_vA = mdot / (v * A(x))
    term1 = v
    term2 = -K * gamma * (mdot_vA ** (gamma - 1)) * (v)

    # Make sure we don't encounter invalid values in power calculations
    try:
            dvdx = (term1 + term2)**(-1) * (K * gamma * mdot_vA ** (gamma - 1) * (v * A(x)) ** (-1) * v * dA(x))
    except Exception as e:
        print(f"Error calculating dvdx: {e}")
        return 0
     
    return dvdx

def system_of_equations_bulk(x, y):
    v, a = y
    mdot_vA = mdot / (v * A(x))
    term1 = mdot_vA * a * v
    term2 = -K * gamma * (mdot_vA ** (gamma - 1)) * ((a * A(x) + v * dA(x)) / (v * A(x)))
    dydx = [a, (1 / eta) * (term1 + term2)]
    return dydx

### find solutions to system of equations ### 
solution_novisc = solve_ivp(system_of_equations_novisc, 
    (0, 10), [v0], t_eval=x_values, 
    dense_output=True, vectorized=True,
    method='LSODA', rtol=1e-4, atol=1e-6)

solution_bulkonly = solve_ivp(lambda t, y: system_of_equations_bulk(t, y), 
        (x0, xf), initial_conditions, 
        method='BDF', dense_output=True, vectorized=True, 
        atol=atol, rtol=rtol)


### Evaluate the solution at various x values ###
y_values_novisc = solution_novisc.sol(x_values)
y_values_bulkonly = solution_bulkonly.sol(x_values)
# y_values_shearonly = solution_shearonly.sol(x_values)
# y_values_full = solution_full.sol(x_values)

### Evaluate the solution ###
v_novisc = y_values_novisc[0]

v_bulkonly = y_values_bulkonly[0]
a_bulkonly = y_values_bulkonly[1]
# %%
### plot velocities ###
fig, ax_vel = plt.subplots()

ax_vel.plot(x_values, v_novisc, label='Only P')
ax_vel.plot(x_values, v_bulkonly, linestyle='--', label='bulk')
ax_vel.text(max(x_values)*(2/3),max(v_bulkonly),r'$\mu$={:.2f}, $\eta$={:.2f}, A=$\sin(x)$+1'.format(mu,eta))
ax_vel.set_xlabel('$x$')
ax_vel.legend()
#%%
### plot area ###
fig,ax_area = plt.subplots()

ax_area.plot(x_values, A(x_values), label='A')
ax_area.set_xlabel('$x$')
ax_area.plot(x_values, dA(x_values), label='dA')
ax_area.legend()
#%%
### save data to plot later ### 
np.save('v_novisc_xA.npy',v_novisc)
np.save('v_bulkonly_xA.npy',v_bulkonly)
# %%
### load data
load_v_novisc_constA = np.load("v_novisc_constA.npy")
load_v_bulkonly_constA = np.load('v_bulkonly_constA.npy')

load_v_novisc_sinA = np.load("v_novisc_sinA.npy")
load_v_bulkonly_sinA = np.load('v_bulkonly_sinA.npy')

load_v_novisc_xA = np.load("v_novisc_xA.npy")
load_v_bulkonly_xA = np.load('v_bulkonly_xA.npy')
#%%
### plot 
fig, ax = plt.subplots(1,3, figsize=(12,5))

ax[0].plot(x_values, load_v_novisc_constA, label='Only P')
ax[0].plot(x_values, load_v_bulkonly_constA, '--', label='Bulk Included')
ax[0].text(max(x_values)/2,1+0.02,r'A=1')
ax[0].legend()

ax[1].plot(x_values, load_v_novisc_sinA, label='Only P')
ax[1].plot(x_values, load_v_bulkonly_sinA, '--', label='Bulk Included')
ax[1].text(max(x_values)/2,max(load_v_bulkonly_sinA),r'A=sin(x) + 1')

ax[2].plot(x_values, load_v_novisc_xA, label='Only P')
ax[2].plot(x_values, load_v_bulkonly_xA, '--', label='Bulk Included')
ax[2].text(max(x_values)/2,max(load_v_bulkonly_xA),r'A=sin(x) + 1')
# %%

