# %%
import numpy as np 
import matplotlib.pyplot as plt
#%%
 ### fully developed laminar flow btw 2 parallel plates
 ### essentially a pipe 
 ### for set x 
x = 1# np.linspace(0,10,100)
y = np.linspace(0,10,100)
h = max(y) # height btw plates 
b = 0
dPx = -1 #-(m*x + b)

def velocityprofile(y,mu):
    U = max(y)
    return U/h*y - 1/(2*mu)*dPx*y*(h-y)
#%%
fig,ax=plt.subplots()
### plotting the vy velocity profile over the height of the pipe
### for varying shear visocisty (mu)
mu_values =np.linspace(1e-1,1,4)
for mu in mu_values:

    velocity = velocityprofile(y, mu)

    ax.plot(velocityprofile(y,mu),y,label=r'$\mu$ = {:.2f}'.format(mu))
    ax.vlines(0,-0.1,max(y),colors='black',linestyles='dashed')
    ax.hlines(max(y),0,max(velocityprofile(y,mu)), colors='black',linestyles='dotted')
    ax.set_ylabel('Height')
    ax.set_xlabel('Velocity Profile in y')
    # ax.text(max(velocityprofile(y,mu))*(5/6), max(y)*(5/6), r'$\mu$ = {:.2f}'.format(mu))
    if mu == 1:
        ax.text(max(velocityprofile(y,min(mu_values))),max(y)-0.06,'h')
        ax.arrow(0,6,max(velocityprofile(y,1))-0.3,0,head_width=0.2)
        ax.arrow(0,8,16-.3,0,head_width=0.2)
        ax.arrow(0,4,16-.3,0,head_width=0.2)
        ax.arrow(0,2,10-.3,0,head_width=0.2)

ax.legend()
# %%
### similarily for circular tube 
def circvelprofile(R):
    return (R**2 - a**2)/(4*mu)*Pz

z = 1
Pz = -(m*z)
R = np.linspace(-10,10,100) #radius
a = max(R)/2

fig,ax=plt.subplots()
ax.plot(circvelprofile(R),R)
# %%
