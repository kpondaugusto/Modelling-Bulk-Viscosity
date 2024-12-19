#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Parameters
rho = 1.0          # density
mu = 0.01          # shear viscosity (increased for stability)
eta = 0.001         # bulk viscosity (increased for stability)
G = -0.1          # pressure gradient (further reduced to stabilize), assuming P decreasing linearly
L = 1            # length of the domain in x-direction
H0 = 0.1           # initial height at x=0
H1 = 0.01          # final height at x=L
Nx = 30            # number of grid points in x-direction
Ny = 30            # number of grid points in y-direction
dx = L / (Nx - 1)  # grid spacing in x-direction
max_iter = 5000    # maximum number of iterations
tol = 1e-6         # tolerance for convergence

# Initialize velocity field (u_x) with small initial values to prevent runaway values
u_x = np.zeros((Nx, Nx))  # small initial velocity (all zeros)
u_x_old = np.copy(u_x)    # previous iteration of u_x

# Small epsilon to prevent division by zero in finite differences
epsilon = 1e-10

# Calculate the x and y coordinates
x = np.linspace(0, L, Nx)

# Finite difference method
for iteration in range(max_iter):
    u_x_old[:] = u_x[:]  # Store the previous velocity field

    # Loop over interior grid points (excluding boundaries)
    for i in range(1, Nx-1):  # loop in x-direction
        # Calculate the height at this x (linearly decreasing from H0 to H1)
        H = max(H0 - (H0 - H1) * (x[i] / L),epsilon)
        
        # Calculate the grid spacing in y-direction (dy) based on height
        dy = max(H / (Ny - 1),epsilon)
        G_dynamic = G * (H0 / H)  # Adjust pressure gradient dynamically
        dt = min(dx**2, dy**2) / (2 * mu)

        # Check for very small dy
        if dy < epsilon:
            print(f"Warning: dy is too small at x={x[i]}. Value: {dy}")
            dy = epsilon  # Prevent zero division

        for j in range(1, Nx-1):  # loop in y-direction
            # shear viscosity terms 
            # Central difference for second derivatives with epsilon added for stability
            duxx = (u_x[j, i+1] - 2 * u_x[j, i] + u_x[j, i-1]) / max(dx**2 + epsilon,epsilon)  # second derivative in x
            duyy = (u_x[j+1, i] - 2 * u_x[j, i] + u_x[j-1, i]) / max(dy**2 + epsilon,epsilon)  # second derivative in y
            # Check for NaNs
            if np.isnan(duxx) or np.isnan(duyy):
                print(f"NaN detected at i={i}, j={j}")
                print(f"u_x[j, i+1]={u_x[j, i+1]}, u_x[j, i]={u_x[j, i]}, u_x[j, i-1]={u_x[j, i-1]}")

            # Print intermediate terms to debug
            # if iteration == 0:  # Only print for the first iteration for debugging
            #     print(f"Iteration {iteration}, Cell ({j}, {i}):")
            #     print(f"duxx: {duxx}, duyy: {duyy}")
            
            # bulk viscosity term
            # convective_term = rho * u_x[j, i] * (u_x[j, i+1] - u_x[j, i-1]) / (2 * dx)  # Just for debugging, set to 0
            # Convective term using upwind differencing
            convective_term = (u_x[j, i] - u_x[j, i-1]) / dx

            # Diffusive term with artificial viscosity
            diffusive_term = (u_x[j+1, i] - 2*u_x[j, i] + u_x[j-1, i]) / dy**2 #duyy 
                    
            # Check for NaN/Inf BEFORE the update
            if np.isnan(convective_term) or np.isinf(convective_term):
                print(f"Convective term invalid at i={i}, j={j}")
                convective_term = 0.0

            # Update equation with artificial viscosity
            u_x[j, i] = (u_x[j, i] + dt * ((-G_dynamic ) + (mu + eta) * duxx + mu * diffusive_term - convective_term))/rho
             # Debugging: Print intermediate terms for a few grid points
            if iteration % 500 == 0 and i == 1 and j == 1:  # print for specific points for debug
                print(f"Iteration {iteration}: i={i}, j={j}")
                print(f"   duxx: {duxx}, convective_term: {convective_term}, u_x: {u_x[j, i]}")

            # The discretized equation with additional epsilon for numerical stability
            # u_x[j, i] = (u_x[j, i] - (G_dynamic * dy**2) / (mu + eta) + (mu + eta) * duxx + mu * duyy - convective_term) / (1 + (mu + eta) * dy**2 / (2 * rho))
                
            # Clip to prevent unphysical blow-ups
            u_x[j, i] = np.clip(u_x[j, i], -1e3, 1e3)
    for i in range(Nx):
        H = max(H0 - (H0 - H1) * (x[i] / L), epsilon)
        u_x[:, i] *= (H0 / H)  # Scale velocities inversely with height

        u_x = np.clip(u_x, -1e3, 1e3)  # Clip values to prevent blow-up

     # Check for convergence
    max_change = np.max(np.abs(u_x - u_x_old))
    if np.isnan(max_change):
        print("NaN encountered in convergence check!")
        break

    if max_change < tol:
        print(f"Converged after {iteration+1} iterations.")
        break

# Debugging: Check the minimum and maximum values of u_x to see if there's variation
print(f"Min velocity: {np.min(u_x)}, Max velocity: {np.max(u_x)}")
#%%
# Create Y based on the height H at each x
Y = np.zeros((Nx, Ny))  # Initialize an array for y-coordinates (varying height)

for i in range(Nx):
    H = H0 - (H0 - H1) * (x[i] / L)  # Calculate the height at this x
    Y[i, :] = np.linspace(0, H, Ny)  # Linearly space y from 0 to H for each x

X, _ = np.meshgrid(x, np.linspace(0, 1, Ny))
# Ensure the shapes of X, Y, and u_x match
print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}, Shape of u_x: {u_x.shape}")
#%%
# Plot the solution
plt.figure(figsize=(8, 6))
cp = plt.contourf(1-X, Y, u_x.T, 20, cmap='viridis')  # Transpose u_x to match X, Y shapes
plt.colorbar(cp)
plt.clabel(cp, inline=1, fontsize=10)
plt.title("Velocity Profile $u_x(x, y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.text(0.5,0.09,r'dP/dy={:.2f}, $\mu$={:.2f}, $\eta$={:.3f}'.format(G, mu, eta))
# plt.legend(loc='upper right')
# plt.legend()
# %%
