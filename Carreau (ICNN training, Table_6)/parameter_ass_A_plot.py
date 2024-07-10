import pickle
import numpy as np
import matplotlib.pyplot as plt
from ICNN import ICNN

RANGE_MULTIPLIER = 2.5
n_points = 200
plot_diff = True

name = '2-0'
sign = 1 if name in ['1-2', '1-6', '2-0'] else -1

# Determine transition value: The last point of log scale is the first point of linear scale
log_values = np.logspace(np.log10(1e-2), np.log10(1e1), n_points // 2, endpoint=False)
lin_values = np.linspace(log_values[-1], 1e2, n_points // 2, endpoint=True)

# Combine both arrays, ensuring they cover the entire range from min_x to max_x without overlap
x = np.sort(np.concatenate([log_values, lin_values]))
min_x = np.min(x)
max_x = np.max(x)
eq2_eps = 1


def k(x):
    return sign * icnn(x).item()  # Ensure k(x) returns a scalar float


def eq2(t, s):
    return np.abs(k(t) * t - k(s) * s)


def eq3(t, s):
    return k(t) * t - k(s) * s


with open(f'ass_A_parameters_{name}.pkl', 'rb') as f:
    alpha, C, r, M = pickle.load(f)

# load the model
with open(f'n{name}.pkl', 'rb') as f:
    icnn = pickle.load(f)

print('Loading')
print('alpha:', alpha)
print('C:', C)
print('r:', r)
print('M:', M)

# Generate t_values and s_values for plotting
n_log = n_points // 2
n_lin = n_points // 2

# Determine transition value: The last point of log scale is the first point of linear scale
log_values = np.logspace(np.log10(min_x), np.log10(max_x * RANGE_MULTIPLIER), n_log, endpoint=False)
lin_values = np.linspace(min_x, max_x * RANGE_MULTIPLIER, n_lin, endpoint=True)

# Combine both arrays, ensuring they cover the entire range from min_x to max_x without overlap
t_values = np.sort(np.concatenate([log_values, lin_values]))

s_values = t_values.copy()

# Equation 3 values and mask
T, S = np.meshgrid(t_values, s_values)

# Calculate k_t
k_t = np.array([k(t) for t in t_values])

# 1D plot
plt.figure()
if not plot_diff:
    plt.plot(t_values, k_t, label='$k(t)$')
    plt.plot(t_values, C * (t_values ** alpha * (1 + t_values) ** (1 - alpha)) ** (r - 2),
             label='$C  (t^{\\alpha}  (1 + t)^{1 - \\alpha})^{r - 2}$')
    # plt.scatter(t_values, k_t, marker='x', color='red')
    plt.title('1D Plot')
else:
    plt.plot(t_values, C * (t_values ** alpha * (1 + t_values) ** (1 - alpha)) ** (r - 2) - k_t)
    plt.title('$C  (t^{\\alpha}  (1 + t)^{1 - \\alpha})^{r - 2} - k(t)$')
    plt.grid()
plt.xlabel('$t$')
plt.ylabel('Values')
plt.legend()

plt.show()

# Equation 2 values and mask
eq2_expr1 = np.vectorize(eq2)(T, S)
eq2_expr2 = C * np.abs(T - S) * ((T + S) ** alpha * (1 + T + S) ** (1 - alpha)) ** (r - 2)
eq2_mask = np.abs(S / T - 1) <= 1
eq2_expr1 = np.where(eq2_mask, eq2_expr1, np.nan)
eq2_expr2 = np.where(eq2_mask, eq2_expr2, np.nan)

# 3D plot for Equation 2
if not plot_diff:
    # 3D scatter plot for non-difference expressions
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(T, S, eq2_expr1, label='$|k_t \cdot t - k_s \cdot s|$', color='blue')
    ax3d.scatter(T, S, eq2_expr2,
                 label='$C \cdot |t - s| \cdot ((t + s)^{\\alpha} \cdot (1 + t + s)^{1 - \\alpha})^{r - 2}$',
                 color='red')
    ax3d.set_xlabel('$t$')
    ax3d.set_ylabel('$s$')
    ax3d.set_zlabel('Values')
    ax3d.legend()
    plt.title('3D Plot for Equation 2')
    plt.show()
else:
    # 2D scatter plot for differences
    fig, ax = plt.subplots()
    differences = eq2_expr2 - eq2_expr1
    scatter = ax.scatter(T, S, c=differences, cmap='viridis', s=5)
    print('Negative values for Equation 2:', differences[differences < -0.])
    ax.scatter(T[differences < -0.], S[differences < -0.], color='red', s=5)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$s$')
    ax.legend()
    plt.title('$C|t - s| ((t + s)^{\\alpha}(1 + t + s)^{1 - \\alpha})^{r - 2} - |k(t)  t - k(s)  s|$')
    plt.show()

# Vectorize the eq3 function
eq3_expr1 = np.vectorize(eq3)(T, S)
eq3_expr2 = M * (T - S) * ((T + S) ** alpha * (1 + T + S) ** (1 - alpha)) ** (r - 2)
eq3_mask = T >= S
eq3_expr1 = np.where(eq3_mask, eq3_expr1, np.nan)
eq3_expr2 = np.where(eq3_mask, eq3_expr2, np.nan)

# 3D plot for Equation 3
fig, ax = plt.subplots()

if not plot_diff:
    # 3D scatter plot for non-difference expressions
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.scatter(T, S, eq3_expr1, label='$k_t \cdot t - k_s \cdot s$', color='blue')
    ax3d.scatter(T, S, eq3_expr2,
                 label='$M \cdot (t - s) \cdot ((t + s)^{\\alpha} \cdot (1 + t + s)^{1 - \\alpha})^{r - 2}$',
                 color='red')
    ax3d.set_xlabel('$t$')
    ax3d.set_ylabel('$s$')
    ax3d.set_zlabel('Values')
    ax3d.legend()
    plt.title('3D Plot for Equation 3')
    plt.show()
else:
    # 2D scatter plot for differences, with color magnitude
    differences = eq3_expr1 - eq3_expr2
    print('Negative values for Equation 3:', differences[differences < -0.])
    scatter = ax.scatter(T, S, c=differences, cmap='viridis', s=5)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$s$')
    ax.legend()
    plt.title('$k(t) t - k(s) s - M  (t - s)((t + s)^{\\alpha}  (1 + t + s)^{1 - \\alpha})^{r - 2}$')
    plt.show()
