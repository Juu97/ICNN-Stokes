import os

# Setting environmental variable for OpenMP
os.environ['OMP_NUM_THREADS'] = '1'

import pickle
from firedrake import *
from ICNN import ICNN
import ufl
from source_term import fx_fy, u_ex, p_ex
from functions import carreau, sobolev_errornorm, lebesgue_errornorm, compute_convergence_order, plot_convergence
import matplotlib.pyplot as plt
import numpy as np

# ----------------- #
# User parameters - #
pol_degree_u = 2
pol_degree_p = 1
r = 1.6

use_icnn = True
viscosity_is_convex = True  # Set it to True/False based on dataset. Required only for ICNN case. Turn it to False for testing ICNN_carreau_r28.pkl
icnn_file = 'ICNN_carreau_r16.pkl'
# ----------------- #

# Load ICNN model if use_icnn is True
if use_icnn:
    with open(icnn_file, 'rb') as f:
        icnn_pytorch = pickle.load(f)

    def get_weights(model):
        weights = []
        biases = []
        for idx, param in enumerate(model.parameters()):
            if idx % 2 == 0:
                weights.append(param.detach().numpy().tolist())

            else:
                biases.append(param.detach().numpy().tolist())

        return weights, biases


    net_weights_list, net_biases_list = get_weights(icnn_pytorch)


    def icnn_viscosity(u, convex_viscosity=True):
        """Compute viscosity using the ICNN model."""
        z = inner(sym(grad(u)), sym(grad(u)))
        z = (z - icnn_pytorch.input_min) / (icnn_pytorch.input_max - icnn_pytorch.input_min)

        def elu(x):
            condition = ufl.ge(x, 0)
            return ufl.conditional(condition, x, ufl.exp(x) - 1)

        num_of_layers = len(net_weights_list)
        output_from_layers = [[] for _ in range(num_of_layers)]

        # first layer
        for idx in range(len(net_weights_list[0])):
            output_from_layers[0].append(elu(net_weights_list[0][idx][0] * z + net_biases_list[0][idx]))

        # middle layers
        for layer_number in range(1, num_of_layers):
            for idx_neuron in range(len(net_weights_list[layer_number])):
                partial_sum = 0
                for idx_previous_neuron_output in range(len(output_from_layers[layer_number - 1])):
                    partial_sum += net_weights_list[layer_number][idx_neuron][idx_previous_neuron_output] * \
                                   output_from_layers[layer_number - 1][idx_previous_neuron_output]

                partial_sum += net_biases_list[layer_number][idx_neuron]

                if layer_number < num_of_layers - 1:
                    output_from_layers[layer_number].append(elu(partial_sum))
                else:
                    output_from_layers[layer_number].append(partial_sum)

        sign = 1 if convex_viscosity else -1
        return sign * (output_from_layers[num_of_layers - 1][0] * (
                icnn_pytorch.output_max - icnn_pytorch.output_min) + icnn_pytorch.output_min)

# Simulation information
print(f'Simulation with ICNN: {use_icnn}. Degree pol u: {pol_degree_u}, degree pol p: {pol_degree_p}. r = {r}.\n')

# Error lists
errors_u_L2, errors_u_H1, errors_p_L2, errors_p_H1, errors_u_Wp, errors_p_Lp = ([] for _ in range(6))

# Conjugate exponent
conjugate_r = r / (r - 1)

# Grid sizes
NN = [8, 16, 32]
for N in NN:
    # Mesh initialization and transformation
    mesh = UnitSquareMesh(N, N)
    mesh.coordinates.dat.data[:, 1] -= 0.5
    mesh.coordinates.dat.data[:, 0] -= 0.5

    # Define function spaces for velocity (V) and pressure (Q)
    V = VectorFunctionSpace(mesh, "CG", pol_degree_u)
    Q = FunctionSpace(mesh, "CG", pol_degree_p)
    W = V * Q

    # Define trial and test functions
    w = Function(W)
    (v, q) = TestFunctions(W)

    # Define source term
    x, y = SpatialCoordinate(mesh)
    f = as_vector((fx_fy(x, y, r)))

    # Define boundary condition
    g = Function(V)
    g.interpolate(as_vector((u_ex(x, y, r))))
    bc = DirichletBC(W.sub(0), g, "on_boundary")

    # Exact solutions
    u_exact = as_vector((u_ex(x, y, r)))
    p_exact = p_ex(x, y, r)

    # Split function for velocity and pressure. Initializing guess on Newtonian case.
    u_initial_guess = project(as_vector((u_ex(x, y, r=2.0))), V)
    p_initial_guess = interpolate(p_ex(x, y, r=2.0), Q)

    w.sub(0).assign(u_initial_guess)
    w.sub(1).assign(p_initial_guess)
    (u, p) = split(w)

    # Weak problem formulation
    viscosity = icnn_viscosity(u, convex_viscosity=viscosity_is_convex) if use_icnn else carreau(u, r)
    F = (2.0 * viscosity * inner(sym(grad(u)), sym(grad(v))) - p * div(v) + div(u) * q) * dx
    L = inner(f, v) * dx
    F = F - L

    # Define null space and Jacobian for solver
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    J = derivative(F, w)

    # Problem and solver definition
    problem = NonlinearVariationalProblem(F, w, bc, J)
    solver = NonlinearVariationalSolver(problem, nullspace=nullspace)

    solver.solve()

    # Rename output for clarity
    u, p = w.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")

    # Error computation
    u_sobolev_p_error = sobolev_errornorm(u_exact, u, r)
    p_lebesgue_pcong_error = lebesgue_errornorm(p_exact, p, conjugate_r)

    # Error reporting
    print(f'ERRORS [GRID {N}x{N}] -------------------------------------------------- #')
    print(f'u || Error L2: {errornorm(u_exact, u)} , Error H1: {errornorm(u_exact, u, norm_type="H1")}')
    print(f'p || Error L2: {errornorm(p_exact, p)} , Error H1: {errornorm(p_exact, p, norm_type="H1")}')
    print(f'u || Error W(1,{r}): {u_sobolev_p_error} , p || Error L({round(conjugate_r, 2)}): {p_lebesgue_pcong_error}')

    # Append errors to lists
    errors_u_L2.append(errornorm(u_exact, u))
    errors_u_H1.append(errornorm(u_exact, u, norm_type="H1"))
    errors_p_L2.append(errornorm(p_exact, p))
    errors_p_H1.append(errornorm(p_exact, p, norm_type="H1"))
    errors_u_Wp.append(u_sobolev_p_error)
    errors_p_Lp.append(p_lebesgue_pcong_error)
    print()

# Compute and print convergence orders

print('CONVERGENCE ORDERS')
print(f'u || Conv. Ord. L2: {compute_convergence_order(errors_u_L2, 2.0)} , Conv. Ord. H1: {compute_convergence_order(errors_u_H1, 2.0)}')
print(f'p || Conv. Ord. L2: {compute_convergence_order(errors_p_L2, 2.0)} , Conv. Ord. H1: {compute_convergence_order(errors_p_H1, 2.0)}')
print(f'u || Conv. Ord. W(1,{r}): {compute_convergence_order(errors_u_Wp, 2.0)} , p || Conv. Ord. L({round(conjugate_r, 2)}): {compute_convergence_order(errors_p_Lp, 2.0)}')

# Plot convergence
errors_dict = {f'u_W(1,{str(round(r, 2))})': errors_u_Wp, f'p_L({str(round(conjugate_r, 2))})': errors_p_Lp}
if use_icnn:
    plot_convergence(NN, errors_dict, plot_fitted_line=False)
else:
    plot_convergence(NN, errors_dict)
