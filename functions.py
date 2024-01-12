from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


def carreau(w, r):
    mu_inf = Constant(0.0)
    mu_0 = Constant(2)
    lam = Constant(1)
    n = r - 1

    mu_eff = mu_inf + (mu_0 - mu_inf) * (
            1 + lam * 2 * (inner(sym(grad(w)), sym(grad(w))))) ** (
                     (n - 1) / 2)
    return mu_eff


def sobolev_errornorm(u_exact, u_h, p):
    diff_grad = grad(u_exact) - grad(u_h)
    integrand_2 = pow(sqrt(inner(diff_grad, diff_grad)), p)

    diff = u_exact - u_h
    integrand_1 = pow(sqrt(inner(diff, diff)), p)

    norm = pow(assemble(integrand_1 * dx) + assemble(integrand_2 * dx), 1 / p)
    return norm


def lebesgue_errornorm(u_exact, u_h, p):
    diff = u_exact - u_h
    integrand_1 = pow(sqrt(inner(diff, diff)), p)

    norm = pow(assemble(integrand_1 * dx), 1 / p)
    return norm


def compute_convergence_order(errors, refinement_factor):
    num_errors = len(errors)
    if num_errors < 2:
        raise ValueError("At least two errors are needed to compute the convergence order.")

    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    log_ratios = np.log(ratios)
    log_refinement = np.log(refinement_factor)

    p = log_ratios / log_refinement

    return round(p.mean(), 2)


def plot_convergence(grid_sizes, errors, plot_fitted_line=True):
    # get h of the grid
    h = []
    for N in grid_sizes:
        mesh = UnitSquareMesh(N, N)
        h_mesh = CellDiameter(mesh)
        h_value = project(h_mesh, FunctionSpace(mesh, "CG", 1)).dat.data[0]
        h.append(h_value)

    # Plot the error versus the grid size on a log-log scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    color_list = ['red', 'red']

    # Plot for u_L2 and u_H1
    ax1.set_xlabel('h')
    ax1.set_ylabel('Error')
    ax1.set_title('Convergence for $u$ in norm $W^{1,r}$')

    errors_set = errors[list(errors.keys())[0]]

    ax1.loglog(h, errors_set, 'o', label="err" + list(errors.keys())[0][1:])

    # Fit a line to the data using linear regression
    coeffs = np.polyfit(np.log(h), np.log(errors_set), 1)
    line = np.exp(coeffs[1]) * h ** coeffs[0]

    # Plot the fitted line
    if plot_fitted_line:
        ax1.loglog(h, line, '-', color=color_list[0])

    # Check the goodness of fit of the line
    r_squared = 1 - np.sum((np.log(errors_set) - np.log(line)) ** 2) / np.sum(
        (np.log(errors_set) - np.mean(np.log(errors_set))) ** 2)

    # Plot h^2, h^3, h^4 lines
    h_values = np.array(h)
    ax1.loglog(h_values, h_values ** 2, '--', label="h = $h^2$", color="darkgrey")
    ax1.loglog(h_values, h_values ** 3, '--', label="h = $h^3$", color="dimgrey")
    ax1.loglog(h_values, h_values ** 4, '--', label="h = $h^4$", color="black")

    ax1.legend(fontsize='large')
    ax1.set(adjustable='box')

    ax2.set_xlabel('h')
    ax2.set_ylabel('Error')
    ax2.set_title('Convergence for $p$ in norm $L^{r\'}$')

    p_names = ['p_Lp']

    errors_set = errors[list(errors.keys())[1]]

    ax2.loglog(h, errors_set, 'o', label="err" + list(errors.keys())[1][1:])

    # Fit a line to the data using linear regression
    coeffs = np.polyfit(np.log(h), np.log(errors_set), 1)
    line = np.exp(coeffs[1]) * h ** coeffs[0]

    # Plot the fitted line
    if plot_fitted_line:
        ax2.loglog(h, line, '-', color=color_list[0])

    # Plot h^2, h^3, h^4 lines
    h_values = np.array(h)

    ax2.loglog(h_values, h_values ** 2, '--', label="h = $h^2$", color="darkgrey")
    ax2.loglog(h_values, h_values ** 3, '--', label="h = $h^3$", color="dimgrey")
    ax2.loglog(h_values, h_values ** 4, '--', label="h = $h^4$", color="black")

    ax2.legend(fontsize='large')
    ax2.set(adjustable='box')

    # Set equal aspect ratio for both axes

    plt.show()
