# %%
import numpy as np 
import matplotlib.pyplot as plt
#%%
 ### fully developed laminar flow btw 2 parallel plates
 ### essentially a pipe 
 ### for set x 
y = 1# np.linspace(0,10,100)
x = np.linspace(0,10,100)
dPy = -1 #-(m*x + b)
eta = 1e-2

def velocityprofile(x,eta):
    xf = max(x)
    uf = max(x) #?
    return -uf/xf * x + 1/(2*eta)*dPy*x*(x-xf)

#%%
fig, ax = plt.subplots()

ax.plot(velocityprofile(x,eta),x)
# ax.vlines(0,-0.1,max(y),colors='black',linestyles='dashed')
# ax.hlines(max(y),0,max(velocityprofile(y)), colors='black',linestyles='dotted')
ax.set_ylabel('Height')
ax.set_xlabel('Velocity Profile in y')
#%%
fig,ax=plt.subplots()
### plotting the vy velocity profile over the height of the pipe
### for varying shear visocisty (mu)
eta_values = np.linspace(1e-1,1,4)
for eta in eta_values:

    ax.plot(velocityprofile(x,eta),x,label=r'$\eta$ = {:.2f}'.format(eta))
    # ax.vlines(0,-0.1,max(x),colors='black',linestyles='dashed')
    # ax.hlines(max(x),0,max(velocityprofile(y,eta)), colors='black',linestyles='dotted')
    ax.set_ylabel('Height')
    ax.set_xlabel('Velocity Profile in y')
    # ax.text(max(velocityprofile(y,mu))*(5/6), max(y)*(5/6), r'$\mu$ = {:.2f}'.format(mu))
ax.legend()
# %%
