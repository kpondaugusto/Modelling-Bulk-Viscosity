#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# Parameters
rho = 1.0          # density
mu = 0.01         # dynamic viscosity (increased for stability)
eta = 0.001        # additional viscosity (increased for stability)
G = -10          # pressure gradient (further reduced to stabilize)
L = 1.0            # length of the domain in x-direction
H0 = 0.1           # initial height at x=0
Nx = 50            # number of grid points in x-direction
Ny = 50            # number of grid points in y-direction
dx = L / (Nx - 1)  # grid spacing in x-direction
dy = H0 / (Ny - 1)  # grid spacing in x-direction
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
        
        # Check for very small dy
        if dy < epsilon:
            print(f"Warning: dy is too small at x={x[i]}. Value: {dy}")
            dy = epsilon  # Prevent zero division

        for j in range(1, Nx-1):  # loop in y-direction
            # Central difference for second derivatives with epsilon added for stability
            duxx = (u_x[j, i+1] - 2 * u_x[j, i] + u_x[j, i-1]) / (dx**2 + epsilon)  # second derivative in x
            duyy = (u_x[j+1, i] - 2 * u_x[j, i] + u_x[j-1, i]) / (dy**2 + epsilon)  # second derivative in y

            # Print intermediate terms to debug
            # if iteration == 0:  # Only print for the first iteration for debugging
            #     print(f"Iteration {iteration}, Cell ({j}, {i}):")
            #     print(f"duxx: {duxx}, duyy: {duyy}")
            
            # Remove convective term temporarily for stability check
            convective_term =  rho * u_x[j, i] * (u_x[j, i+1] - u_x[j, i-1]) / (2 * dx)  # Just for debugging, set to 0
            
             # Debugging: Print intermediate terms for a few grid points
            if iteration % 500 == 0 and i == 1 and j == 1:  # print for specific points for debug
                print(f"Iteration {iteration}: i={i}, j={j}")
                print(f"   duxx: {duxx}, convective_term: {convective_term}, u_x: {u_x[j, i]}")


            # The discretized equation with additional epsilon for numerical stability
            u_x[j, i] = (u_x[j, i] - (G * dy**2) / (mu + eta) + (mu + eta) * duxx + mu * duyy - convective_term) / (1 + (mu + eta) * dy**2 / (2 * rho))

    # Apply a safety factor to clip the entire velocity field to a reasonable range
    u_x = np.clip(u_x, -1.0, 100.0)

    # Convergence check (based on the change in the velocity field)
    max_change = np.max(np.abs(u_x - u_x_old))
    if max_change < tol:
        print(f"Converged after {iteration+1} iterations.")
        break

# Debugging: Check the minimum and maximum values of u_x to see if there's variation
print(f"Min velocity: {np.min(u_x)}, Max velocity: {np.max(u_x)}")
#%%
X, Y = np.meshgrid(x, np.linspace(0, H0, Ny))
# Ensure the shapes of X, Y, and u_x match
print(f"Shape of X: {X.shape}, Shape of Y: {Y.shape}, Shape of u_x: {u_x.shape}")
#%%
### save data
np.save(".npy")
#%%
# Plot the solution
# X, Y = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, H0, Ny))
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, u_x, 20, cmap='viridis')
plt.colorbar(cp)
plt.title("Velocity Profile $u_x(x, y)$")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# %%
