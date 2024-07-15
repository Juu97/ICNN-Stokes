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

set_log_level("ERROR")

# ----------------- #
# User parameters - #
pol_degree_u = 2
pol_degree_p = 1
r = 2.0

use_icnn = True
icnn_file = 'networks/n'+str(r)[0]+str(r)[2]+'.pkl'

if r > 2:
    viscosity_is_convex = False
else:
    viscosity_is_convex = True
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
        z = ufl.sqrt(inner(sym(grad(u)), sym(grad(u))))
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
errors_u_Wp, errors_p_Lp = ([] for _ in range(2))

# Grid sizes
NN = [2, 4, 8, 16, 32, 64]
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

    parameters = {'ksp_type': 'gmres', 'pc_type': 'gamg',
                  'snes_type': 'newtonls',
                  #'snes_monitor': None,
                   'ksp_rtol': 1e-7,
                   'ksp_atol': 1e-7,
                   'snes_stol': 1e-10,
                   'snes_rtol': 1e-10,
                   'snes_max_it': 100
                   }
        
    # Weak problem formulation
    viscosity = icnn_viscosity(u, convex_viscosity=viscosity_is_convex) if use_icnn else carreau(u, r)
    F = (2.0 * viscosity * inner(sym(grad(u)), sym(grad(v))) - p * div(v) + div(u) * q) * dx
    L = inner(f, v) * dx
    F = F - L

    # Define null space and Jacobian for solver
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
    solve(F == 0, w, bcs=bc, nullspace=nullspace, solver_parameters=parameters)

    # Rename output for clarity
    u, p = w.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")
    
    # Conjugate exponent
    conjugate_r = r / (r - 1)
    if r<2:
        su = r; sp = 2
    else:
        su = 2; sp = conjugate_r

    # Error computation
    u_error = sobolev_errornorm(u_exact, u, su)
    p_error = lebesgue_errornorm(p_exact, p, sp)

    # Error reporting
    print(f'        GRID {N:3d}x{N:3d}: u || Error W(1,{su:3.2f}): {u_error:15g} , p || Error L({sp:3.2f}): {p_error:15g}')

    # Append errors to lists
    errors_u_Wp.append(u_error)
    errors_p_Lp.append(p_error)

# Plot convergence
errors_dict = {f'u_W(1,{str(round(su, 2))})': errors_u_Wp, f'p_L({str(round(sp, 2))})': errors_p_Lp}
if use_icnn:
    plot_convergence(NN, errors_dict, plot_fitted_line=False)
else:
    plot_convergence(NN, errors_dict)
