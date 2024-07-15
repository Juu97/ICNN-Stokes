import os
import sys
import csv

os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import time
from firedrake.petsc import PETSc
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np  # Extract velocity and pressure solutions
from mpl_toolkits.mplot3d import Axes3D
from source_term import fx_fy, u_ex, p_ex
from functions import *

set_log_level("ERROR")


## ----------------------------- CONVERGENCE TEST ------------------------------ ##

file = open("order.csv",'w')
writer = csv.writer(file)
writer.writerow(["r", "j", "Velocity error", "Pressure error"])

for pol_degree_u in range(2,6):
    print(f'P{pol_degree_u}/P{pol_degree_u-1}')
    for r in [1.2, 1.6, 2.0, 2.4, 2.8]:
        print(f'    r = {r}')
        pol_degree_p = pol_degree_u - 1
        
        errors_u_Wp = []
        errors_p_Lp = []
        
        NN = [8,16]
        for N in NN:
            mesh = UnitSquareMesh(N, N)
            mesh.coordinates.dat.data[:, 1] -= 0.5
            mesh.coordinates.dat.data[:, 0] -= 0.5
        
            # Define mixed function space for velocity and pressure
            V = VectorFunctionSpace(mesh, "CG", pol_degree_u)
            Q = FunctionSpace(mesh, "CG", pol_degree_p)
            W = V * Q
        
            # Define trial and test functions
            w = Function(W)
            (v, q) = TestFunctions(W)
        
            # Define source term
            x, y = SpatialCoordinate(mesh)
        
            f = as_vector((fx_fy(x, y, r)))
        
            # Boundary condition
            g = Function(V)
            g.interpolate(as_vector((u_ex(x, y, r))))
            bc = DirichletBC(W.sub(0), g, "on_boundary")
        
            u_exact = as_vector((u_ex(x, y, r)))
            p_exact = p_ex(x, y, r)
        
            (u, p) = split(w)
        
            # WEAK PROBLEM WITH EXPLICIT NU
            F = (2.0 * carreau(u, r) * inner(sym(grad(u)), sym(grad(v))) - p * div(v) + div(u) * q) * dx
            L = inner(f, v) * dx
        
            F = F - L
            nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
        
            parameters = {'snes_type': 'newtontr',
                          'snes_tr_fallback_type': 'dogleg', 
                          #'snes_monitor': None,
                           'ksp_rtol': 1e-14,
                           'snes_stol': 1e-10,
                           'snes_rtol': 1e-10,
                           'snes_max_it': 1000
                           }
        
            solve(F == 0, w, bcs=bc, nullspace=nullspace, solver_parameters=parameters)
        
            u, p = w.subfunctions
            u.rename("Velocity")
            p.rename("Pressure")
        
            conjugate_r = r / (r-1)

            if r<2:
                su = r; sp = 2.
            else:
                su = 2.; sp = conjugate_r
        
            u_error = sobolev_errornorm(u_exact, u, su)
            p_error = lebesgue_errornorm(p_exact, p, sp)
        
            print(f'        GRID {N:3d}x{N:3d}: u || Error W(1,{su:3.2f}): {u_error:15g} , p || Error L({sp:3.2f}): {p_error:15g}')
        
            errors_u_Wp.append(u_error)
            errors_p_Lp.append(p_error)
        
        # ORDER OF CONVERGENCE
        print(f'        ORDER: u || Conv. Ord. W(1,{round(su,2)}): {compute_convergence_order(errors_u_Wp, 2.00)} ,  p || Conv. Ord. L({round(sp,2)}): {compute_convergence_order(errors_p_Lp, 2.0)}')
        
        writer.writerow([r, pol_degree_u, 
                         compute_convergence_order(errors_u_Wp, 2.0),
                         compute_convergence_order(errors_p_Lp, 2.0)])

file.close()
