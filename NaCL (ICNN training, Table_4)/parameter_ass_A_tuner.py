import pickle
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from ICNN import ICNN
import matplotlib.pyplot as plt

RANGE_MULTIPLIER = 2.5
n_points = 200
do_plt = True


# Load data

name = '00'

df = pd.read_csv(f"flow_data_{name}.csv")
x = df["shear_rate"].to_numpy()
y = df["viscosity"].to_numpy()
min_x = np.min(x)
max_x = np.max(x)
eq2_eps = 1

# load the model

with open(f'icnn_model_{name}.pkl', 'rb') as f:
    icnn = pickle.load(f)


def k(x):
    return icnn(x).item()  # Ensure k(x) returns a scalar float


def penalty(x):
    return np.where(x >= 0, x, 1e9)


def eq2(t, s):
    return np.abs(k(t) * t - k(s) * s)


def eq3(t, s):
    return k(t) * t - k(s) * s


# Vectorize the eq2 and eq3 functions
vectorized_eq2 = np.vectorize(eq2)
vectorized_eq3 = np.vectorize(eq3)


def objective(params):
    alpha, C, r, M = params

    n_log = n_points // 2
    n_lin = n_points // 2

    # Determine transition value: The last point of log scale is the first point of linear scale
    log_values = np.logspace(np.log10(min_x), np.log10(max_x * RANGE_MULTIPLIER), n_log, endpoint=False)
    lin_values = np.linspace(min_x, max_x * RANGE_MULTIPLIER, n_lin, endpoint=True)

    # Combine both arrays, ensuring they cover the entire range from min_x to max_x without overlap
    t_values = np.sort(np.concatenate([log_values, lin_values]))
    s_values = t_values.copy()

    T, S = np.meshgrid(t_values, s_values)

    k_t = np.array([k(t) for t in t_values])

    # Equation 1
    expr1 = -k_t + C * (t_values ** alpha * (1 + t_values) ** (1 - alpha)) ** (r - 2)
    total_penalty1 = penalty(expr1).sum()

    # Equation 2
    eq2_expr1 = vectorized_eq2(T, S)
    eq2_expr2 = C * np.abs(T - S) * ((T + S) ** alpha * (1 + T + S) ** (1 - alpha)) ** (r - 2)
    mask2 = np.abs(S / T - 1) <= eq2_eps
    total_penalty2 = penalty(eq2_expr2[mask2] - eq2_expr1[mask2]).sum()

    # Equation 3
    eq3_expr1 = vectorized_eq3(T, S)
    eq3_expr2 = M * (T - S) * ((T + S) ** alpha * (1 + T + S) ** (1 - alpha)) ** (r - 2)
    mask3 = T >= S
    total_penalty3 = penalty(eq3_expr1[mask3] - eq3_expr2[mask3]).sum()

    total_penalty = total_penalty1 + total_penalty2 + total_penalty3
    return total_penalty


if __name__ == '__main__':
    print('Loading', name)

    if do_plt:
        n_log = n_points // 2
        n_lin = n_points // 2

        # Determine transition value: The last point of log scale is the first point of linear scale
        log_values = np.logspace(np.log10(min_x), np.log10(max_x), n_log, endpoint=False)
        lin_values = np.linspace(log_values[-1], max_x, n_lin, endpoint=True)

        # Combine both arrays, ensuring they cover the entire range from min_x to max_x without overlap
        t_values = np.sort(np.concatenate([log_values, lin_values]))
        val = [k(t) for t in t_values]
        plt.loglog(t_values, val)
        plt.scatter(t_values, val)
        plt.show()

    # Define the bounds for each parameter
    bounds = [(0, 1), (0, 50), (1, 10), (0, 50)]

    # Run the differential evolution optimizer
    print('Starting optimization')
    result = differential_evolution(objective, bounds, maxiter=250, disp=True, workers=4, tol=1e-6, seed=42)

    print("Optimized parameters:", result.x)

    # Save the optimized parameters with first available name of parameters_X.pkl
    with open(f'ass_A_parameters_{name}.pkl', 'wb') as f:
        pickle.dump(result.x, f)
