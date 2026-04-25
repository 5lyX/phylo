import numpy as np
import matplotlib.pyplot as plt

def main(plot = False):
    # Parameters
    l = 1.45  # Length of the string (m)
    g = 9.81  # Gravity (m/s^2)
    m = 5.0   # Mass (kg)
    dt = 0.001  # Time step (s)
    T_total = 2.0  # Total simulation time (s)
    g_vec = np.array([0, 0, -g])

    # Initial conditions
    theta = np.radians(60)
    r = np.array([l * np.sin(theta), 0.0, -l * np.cos(theta)])  # shifted so z=0 at taut
    v = np.array([-0.36, 0.78, 0.0])
    t = 0.0

    # Storage
    times = []
    positions = []
    velocities = []
    accelerations = []
    tensions = []

    def compute_acceleration(r, v):
        r_norm = np.linalg.norm(r)
        if r_norm >= l - 1e-6:
            # Taut: constraint active
            r_hat = r / r_norm
            lam = (np.dot(r, g_vec) + np.dot(v, v)) / l**2
            a = g_vec - lam * r_hat
            T = lam * m
        else:
            # Slack: free fall
            a = g_vec
            T = 0.0
        return a, T

    # RK4 Integration loop
    while t < T_total:
        # Handle tautening: enforce velocity projection if becoming taut
        r_norm = np.linalg.norm(r)
        if r_norm >= l - 1e-6:
            r_hat = r / r_norm
            v_radial = np.dot(v, r_hat)
            if v_radial > 0:
                v -= v_radial * r_hat

        # RK4 Step
        def f(state):
            r_local, v_local = state[:3], state[3:]
            a_local, _ = compute_acceleration(r_local, v_local)
            return np.hstack((v_local, a_local))

        state = np.hstack((r, v))
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        r = state_next[:3]
        v = state_next[3:]
        a, T = compute_acceleration(r, v)

        # Store data
        times.append(t)
        positions.append(r)
        velocities.append(v)
        accelerations.append(a)
        tensions.append(T)

        t += dt

    # Convert to numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    tensions = np.array(tensions)
    times = np.array(times)

    if plot:
        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        # Velocity
        axs[0, 0].plot(times, velocities[:, 0], label='vx')
        axs[0, 0].plot(times, velocities[:, 1], label='vy')
        axs[0, 0].plot(times, velocities[:, 2], label='vz')
        axs[0, 0].set_ylabel('Velocity (m/s)')
        axs[0, 0].legend()
        axs[0, 0].set_title('Velocity')

        # Acceleration
        axs[0, 1].plot(times, accelerations[:, 0], label='ax')
        axs[0, 1].plot(times, accelerations[:, 1], label='ay')
        axs[0, 1].plot(times, accelerations[:, 2], label='az')
        axs[0, 1].set_ylabel('Acceleration (m/s²)')
        axs[0, 1].legend()
        axs[0, 1].set_title('Acceleration')

        # Position
        axs[1, 0].plot(times, positions[:, 0], label='rx')
        axs[1, 0].plot(times, positions[:, 1], label='ry')
        axs[1, 0].plot(times, positions[:, 2], label='rz')
        axs[1, 0].set_ylabel('Position (m)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].legend()
        axs[1, 0].set_title('Position')

        # Tension
        axs[1, 1].plot(times, tensions, label='Tension', color='purple')
        axs[1, 1].set_ylabel('Tension (N)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].legend()
        axs[1, 1].set_title('Tension')

        plt.tight_layout()
        plt.show()

    return (
        positions,
        velocities,
        accelerations,
        tensions,
    )
