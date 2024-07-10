import pickle
import numpy as np
from matplotlib import pyplot as plt


# Carreau (ICNN training, Table_6) function definition
def carreau(x, n):
    return mu_inf + (mu_0 - mu_inf) * (1 + lambd * x ** 2) ** ((n - 2) / 2)


# Scale function definition
# def scale(value, min_value, max_value):
#     # return (value - min_value) / (max_value - min_value)
#     return value


# Parameters
mu_inf = 0.001
mu_0 = 1
lambd = 100

names = ['1-2', '1-6', '2-4', '2-8']
nn = [1.2, 1.6, 2.4, 2.8]

# Generate x values
min_x = 0.001
max_x = 70

log_values = np.logspace(np.log10(min_x), np.log10(max_x), 5000, endpoint=False)
lin_values = np.linspace(log_values[-1], max_x, 5000, endpoint=True)
x_values = np.sort(np.concatenate([log_values, lin_values]))

for name, n in zip(names, nn):
    with open(f'n{name}.pkl', 'rb') as f:
        icnn = pickle.load(f)

    sign = 1 if n <= 2 else -1

    # Evaluate icnn function at x values
    icnn_values = np.array([sign * icnn(x) for x in x_values])

    # Evaluate Carreau (ICNN training, Table_6) function at x values
    carreau_values = carreau(x_values, n)

    combined_min = min(np.min(icnn_values), np.min(carreau_values))
    combined_max = max(np.max(icnn_values), np.max(carreau_values))

    # Scale icnn and Carreau (ICNN training, Table_6) values
    # icnn_values = scale(icnn_values, combined_min, combined_max)
    icnn_values = icnn_values.reshape((-1,))
    # carreau_values = scale(carreau_values, combined_min, combined_max)

    # Compute absolute differences
    abs_diff = np.abs(icnn_values - carreau_values)

    # Find the maximum absolute difference and its corresponding x value
    l_inf_norm = np.max(abs_diff)
    x_at_l_inf = x_values[np.argmax(abs_diff)]

    print(f'L-infinity difference for n={n}: {l_inf_norm:.4f}')
