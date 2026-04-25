import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt

from ipdb import set_trace as st

def main(plot = False):
    g = 9.81 # Acceleration due to gravity

    # Time setup
    t0, tf, dt = 0, 2, 1e-3
    t_eval = np.arange(t0, tf + dt, dt)

    # Initial conditions
    x0 = 5 * np.sqrt(3)
    x_dot0 = 0
    z0 = [x0, x_dot0]

    from scipy.interpolate import interp1d

    # Pre-allocate variable storage
    n_steps = len(t_eval)
    a1_vals = np.empty(n_steps)
    T1_vals = np.empty(n_steps)
    T2_vals = np.empty(n_steps)
    T3_vals = np.empty(n_steps)
    T4_vals = np.empty(n_steps)
    y_vals  = np.empty(n_steps)

    # Store output values via dense evaluation
    def dae_rhs_factory(storage):
        def dae_rhs(t, z):
            x, x_dot = z
            y = 5 + 5 * np.sqrt(3) - x

            def equations(vars):
                # These equations resulted in negative T4. Therefore T4 is slack.
                a1, T1, T2, T3, T4, x_ddot = vars
                eq1 = 6 + g/2 - T1 - (5/2)*a1
                eq2 = T1 - T2 - a1
                eq3 = T2 - T3 - (9/4)*a1 - g*np.sqrt(3)/4
                eq4 = a1 + x_ddot * (x / np.sqrt(x**2 + 25))
                eq5 = T3*x/np.sqrt(x**2 + 25) - T4*y/np.sqrt(y**2 + 75) + 4*x_ddot
                eq6 = T4 + g/2 + x_ddot * (y/x) * (np.sqrt(x**2 + 25) / np.sqrt(y**2 + 75))

                # Assume T4 is slack
                a1, T1, T2, T3, T4, x_ddot = vars
                eq1 = 6 + g/2 - T1 - (5/2)*a1
                eq2 = T1 - T2 - a1
                eq3 = T2 - T3 - (9/4)*a1 - g*np.sqrt(3)/4
                eq4 = a1 + x_ddot * (x / np.sqrt(x**2 + 25))
                eq5 = T3*x/np.sqrt(x**2 + 25) + 4*x_ddot
                eq6 = T4

                return [eq1, eq2, eq3, eq4, eq5, eq6]

            guess = [0, 0, 0, 0, 0, 0]
            sol = root(equations, guess)
            if not sol.success:
                raise RuntimeError(f"DAE root solver failed at t={t}, x={x}, x_dot={x_dot}")
            
            a1, T1, T2, T3, T4, x_ddot = sol.x
            storage['last'] = (a1, T1, T2, T3, T4, y)  # store current result for retrieval
            return [x_dot, x_ddot]
        
        return dae_rhs

    # Use dict to store most recent algebraic solution
    storage = {}
    dae_rhs_wrapped = dae_rhs_factory(storage)

    # Solve
    sol = solve_ivp(dae_rhs_wrapped, [t0, tf], z0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)

    # Extract algebraic variables post-hoc
    for i, (t_i, x_i, x_dot_i) in enumerate(zip(sol.t, sol.y[0], sol.y[1])):
        y_i = 5 + 5 * np.sqrt(3) - x_i

        def equations(vars):
            # These equations resulted in negative T4. Therefore T4 is slack.
            a1, T1, T2, T3, T4, x_ddot = vars
            eq1 = 6 + g/2 - T1 - (5/2)*a1
            eq2 = T1 - T2 - a1
            eq3 = T2 - T3 - (9/4)*a1 - g*np.sqrt(3)/4
            eq4 = a1 + x_ddot * (x_i / np.sqrt(x_i**2 + 25))
            eq5 = T3*x_i/np.sqrt(x_i**2 + 25) - T4*y_i/np.sqrt(y_i**2 + 75) + 4*x_ddot
            eq6 = T4 + g/2 + x_ddot * (y_i/x_i) * (np.sqrt(x_i**2 + 25) / np.sqrt(y_i**2 + 75))
            
            # Assume T4 is slack
            a1, T1, T2, T3, T4, x_ddot = vars
            eq1 = 6 + g/2 - T1 - (5/2)*a1
            eq2 = T1 - T2 - a1
            eq3 = T2 - T3 - (9/4)*a1 - g*np.sqrt(3)/4
            eq4 = a1 + x_ddot * (x_i / np.sqrt(x_i**2 + 25))
            eq5 = T3*x_i/np.sqrt(x_i**2 + 25) + 4*x_ddot
            eq6 = T4
            
            return [eq1, eq2, eq3, eq4, eq5, eq6]

        guess = [0, 0, 0, 0, 0, 0]
        sol_i = root(equations, guess)
        if not sol_i.success:
            raise RuntimeError(f"DAE root solver failed at t={t_i}, x={x_i}")
        
        a1, T1, T2, T3, T4, _ = sol_i.x
        a1_vals[i] = a1
        T1_vals[i] = T1
        T2_vals[i] = T2
        T3_vals[i] = T3
        T4_vals[i] = T4
        y_vals[i] = y_i

    # Time series
    t_series = sol.t
    x_series = sol.y[0]
    x_dot_series = sol.y[1]
    y_series = np.array(y_vals)
    a1_series = np.array(a1_vals)
    T1_series = np.array(T1_vals)
    T2_series = np.array(T2_vals)
    T3_series = np.array(T3_vals)
    T4_series = np.array(T4_vals)

    if plot:
        # Plotting
        var_list = [
            ("x", x_series),
            ("x_dot", x_dot_series),
            ("y", y_series),
            ("a1", a1_series),
            ("T1", T1_series),
            ("T2", T2_series),
            ("T3", T3_series),
            ("T4", T4_series),
        ]

        fig, axs = plt.subplots(len(var_list), 1, figsize=(8, 2.5 * len(var_list)), sharex=True)
        for ax, (label, data) in zip(axs, var_list):
            ax.plot(t_series, data)
            ax.set_ylabel(label)
            ax.grid(True)

        axs[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    return {
        "t": t_series,
        "x": x_series,
        "x_dot": x_dot_series,
        "y": y_series,
        "a1": a1_series,
        "T1": T1_series,
        "T2": T2_series,
        "T3": T3_series,
        "T4": T4_series
    }

if __name__ == "__main__":
    main(plot=True)