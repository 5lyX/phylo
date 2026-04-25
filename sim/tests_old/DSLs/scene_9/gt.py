import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from ipdb import set_trace as st

# Constants
L = 1.0
g = 9.81
theta0 = np.pi / 6
omega0 = 0.0
mu = 0.6
yB = mu * L * np.sin(theta0)

# Residual function from given ODE
def residual(alpha, theta, omega):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    term = (
        (
            4 * L**2 * alpha +
            12 * L * alpha * yB / sin_theta +
            6 * L * g * cos_theta -
            6 * L * omega * yB * cos_theta / sin_theta**2 +
            12 * alpha * yB**2 / sin_theta**4 -
            24 * omega * yB**2 * cos_theta / sin_theta**5
        ) / 12
    )
    return term

# ODE function
def theta_ode(t, state):
    theta, omega = state

    def root(alpha):
        return residual(alpha, theta, omega)

    # Solve for alpha numerically
    alpha = fsolve(root, 0.0)[0]
    return [omega, alpha]

# Initial conditions
initial_state = [theta0, omega0]
t_span = (0, 2)
t_eval = np.arange(t_span[0], t_span[1] + 0.001, 0.001)

# Integrate the ODE
sol = solve_ivp(theta_ode, t_span, initial_state, t_eval=t_eval, rtol=1e-8, atol=1e-8)

# Given: theta(t), omega(t), alpha(t), t
# Replace these with your simulated data
# Example placeholders:
theta = sol.y[0]
omega = sol.y[1]
alpha = np.gradient(omega, t_eval)  # If alpha wasn't stored during simulation
t = sol.t

# Compute positions
xA = -yB / np.tan(theta)
x = xA + (L / 2) * np.sin(theta)
y = (L / 2) * np.sin(theta)

# Compute velocities
vx = (L / 2) * omega * np.cos(theta) + yB * omega / np.sin(theta)**2
vy = (L / 2) * omega * np.cos(theta)

# Compute accelerations
ax = (L / 2) * alpha * np.cos(theta) - (L / 2) * omega**2 * np.sin(theta) + \
     yB * (alpha * np.sin(theta)**2 - 2 * omega**2 * np.cos(theta)) / np.sin(theta)**3
ay = (L / 2) * alpha * np.cos(theta) - (L / 2) * omega**2 * np.sin(theta)

# Find the index where y is closest to yB/2
target_y = yB / 2
idx = np.argmin(np.abs(y - target_y))

# Prune all arrays to that timestep
x = x[:idx+1]
y = y[:idx+1]
vx = vx[:idx+1]
vy = vy[:idx+1]
ax = ax[:idx+1]
ay = ay[:idx+1]
xA = xA[:idx+1]
theta = theta[:idx+1]
omega = omega[:idx+1]
alpha = alpha[:idx+1]
t = t[:idx+1]

# x = x - xA[0]
# xA = xA - xA[0]

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(4, 2, 1)
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.title("Position")
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(t, vx, label='vx')
plt.plot(t, vy, label='vy')
plt.title("Velocity")
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(t, ax, label='ax')
plt.plot(t, ay, label='ay')
plt.title("Acceleration")
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(t, xA, label='xA')
plt.title("xA Position")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, theta, label='theta')
plt.plot(t, omega, label='omega')
plt.plot(t, alpha, label='alpha')
plt.title("Angular Motion")
plt.legend()

plt.tight_layout()
plt.show()