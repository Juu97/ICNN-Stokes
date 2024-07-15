from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


def carreau(w, n):
    mu_inf = Constant(0.001)
    mu_0 = Constant(1)
    lam = Constant(100)

    mu_eff = mu_inf + (mu_0 - mu_inf) * (1 + lam * inner(sym(grad(w)), sym(grad(w)))) ** (
                     (n - 2) / 2)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    color_list = ['red', 'red']

    # Plot for u_L2 and u_H1
    ax1.set_xlabel('h')
    ax1.set_ylabel('Error')

    errors_set = errors[list(errors.keys())[0]]

    ax1.loglog(h, errors_set, '-o', color='red', label="Velocity Error")

    h_values = np.array(h)
    ax1.loglog(h_values, 2 * h_values ** 2, '--', label="$h^2$", color="darkgrey")

    ax1.legend(loc='upper left',fontsize=14)

    ax1.grid()
    ax1.set_xscale('log')
    ax1.set_xticks([0.01, 0.1])

    ax1.set_yscale('log')
    ax1.set_yticks([0.0001, 0.001, 0.01, 0.1, 1])
    ax1.set_ylim([1e-4,1e1])

    ax2.set_xlabel('h')
    ax2.set_ylabel('Error')

    p_names = ['p_Lp']

    errors_set = errors[list(errors.keys())[1]]

    ax2.loglog(h, errors_set, '-o', color='red', label="Pressure Error")

    h_values = np.array(h)

    ax2.loglog(h_values, 0.05 * h_values ** 2, '--', label="$h^2$", color="darkgrey")

    ax2.legend(loc='upper left',fontsize=14)

    ax2.grid()
    ax2.set_xscale('log')
    ax2.set_xticks([0.01, 0.1])

    ax2.set_yscale('log')
    ax2.set_yticks([0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
    ax2.set_ylim([1e-6,1e1])


    # Set equal aspect ratio for both axes
    #plt.savefig('conv.png')

    plt.show()

