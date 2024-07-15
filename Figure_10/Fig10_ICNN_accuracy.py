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
r = 1.6
viscosity_is_convex = True

icnn_files = ['networks/n1-6___800___0-00083207.pkl',
              'networks/n1-6___1600___0-00020291.pkl',
              'networks/n1-6___3200___7-772e-05.pkl',
              'networks/n1-6___50000___4-599e-05.pkl']

epochs = [800, 1600, 3200, 50000] 
validation_error = [0.00083207, 0.00020291, 7.772e-05, 4.599e-05] 

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

def get_weights(model):
    weights = []
    biases = []
    for idx, param in enumerate(model.parameters()):
        if idx % 2 == 0:
            weights.append(param.detach().numpy().tolist())

        else:
            biases.append(param.detach().numpy().tolist())

    return weights, biases

def plot_accuracy_vs_loss(N, loss, errU, errP):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    plt.rcParams["text.usetex"] = True
    
    # Plot for u_L2 and u_H1
    ax1.set_xlabel('Loss on validation set', fontsize=18)
    ax1.set_ylabel(r'$\varepsilon_{ICNN,u}$', fontsize=18)
    ax1.plot(loss, errU, '-', color='black')
    ax1.scatter(loss, errU)
    ax1.set_xlim([0,1e-3])
    ax1.grid()
    
    for i, txt in enumerate(N):
        ax1.annotate(txt, xy=(loss[i], errU[i]), xytext=(loss[i]+0.000015, errU[i]-0.0035), fontsize=14)
    
    ax2.set_xlabel('Loss on validation set', fontsize=18)
    ax2.set_ylabel(r'$\varepsilon_{ICNN,p}$', fontsize=18)
    ax2.plot(loss, errP, '-', color='black')
    ax2.scatter(loss, errP)
    ax2.set_xlim([0,1e-3])
    ax2.grid()
    
    for i, txt in enumerate(N):
        print(txt)
        ax2.annotate(str(txt), xy=(loss[i], errP[i]), xytext=(loss[i]+0.000015, errP[i]-0.001), fontsize=14)
    
    # Set equal aspect ratio for both axes
    # plt.savefig('conv.png')
    
    plt.show()

errU = []
errP = []

for i, icnn_file in enumerate(icnn_files):
    with open(icnn_file, 'rb') as f:
        icnn_pytorch = pickle.load(f)
    
    net_weights_list, net_biases_list = get_weights(icnn_pytorch)
    
    # Grid sizes
    N = 32
        
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
    viscosity = icnn_viscosity(u, convex_viscosity=viscosity_is_convex)
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
   
    errU.append(u_error)
    errP.append(p_error)

    # Error reporting
    print(f'Epochs {epochs[i]}  Validation error {validation_error[i]}: u || Error W(1,{su:3.2f}): {u_error:15g} , p || Error L({sp:3.2f}): {p_error:15g}')

plot_accuracy_vs_loss(epochs, validation_error, errU, errP)
