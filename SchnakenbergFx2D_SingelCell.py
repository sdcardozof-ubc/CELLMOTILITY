"""
This script performs ESFEM simulation under different conditions for Schnakenber model
as kinec of reaction.

Author: Sergio Daniel Cardozo
Date: March 11, 2025
Version: 7.0

Description:
- Compute the mesh coordinates evolving over time forced by different factos, mainly mean curvcature.
- Outputs xdmf file readable in praview, an plots of area, curvature and lambda.

Usage:
Run this script using Python 3.9 or later:
    python3 SchnakenbergFx2D_SinleCell.py
"""

import os
import time
import numpy as np
from mpi4py import MPI  # type: ignore
from petsc4py import PETSc  # type: ignore

# Matplotlib setup
import matplotlib as mpl  # type: ignore
mpl.use('Agg')
import matplotlib.pyplot as plt  # type: ignore

# FEniCSx and UFL imports
import dolfinx  # type: ignore
import basix.ufl  # type: ignore
import ufl  # type: ignore
from basix.ufl import element, mixed_element  # type: ignore
from dolfinx import mesh, log, default_real_type  # type: ignore
from dolfinx.fem import (
    Function, functionspace, Expression, form, Constant  # type: ignore
)
from dolfinx.fem.petsc import (
    NonlinearProblem, assemble_matrix, assemble_vector, create_vector  # type: ignore
)
from dolfinx.nls.petsc import NewtonSolver  # type: ignore
from dolfinx.io import XDMFFile, gmshio  # type: ignore
from ufl import dx, grad, inner, div, dot  # type: ignore

try:
    import gmsh  # type: ignore
except ImportError:
    import sys
    print("This program requires gmsh to be installed")
    sys.exit(0)

def order_dofs(dof):
    """Sort the dofs in order.
    """

    len,_= dof.shape
    # print(type(dof))
    dof_ordered = np.empty_like(dof)
    dof_ordered[0,:] = dof[0,:]
    
    for i in range(len):
        pos = 0
        for j in range(len):
            if dof[j,0] == dof_ordered[i,1]:
                pos = j
                break
        if pos != 0:
            dof_ordered[i+1,:] = dof[pos,:]     
            
    return dof_ordered

def area(dofs,coordinates):
    """Compute the area with the dofs and coordinates.
    """

    num_ele, _ = dofs.shape

    area = 0
    count = 0

    for i in range(num_ele):
        x1 = dofs[i,0]
        x2 = dofs[i,2]
        x3 = dofs[i,1]        
        area += coordinates[x1,0]*coordinates[x2,1] - coordinates[x1,1]*coordinates[x2,0]
        area += coordinates[x2,0]*coordinates[x3,1] - coordinates[x2,1]*coordinates[x3,0]

    area  = 1/2 * abs(area)

    return area

def lambda_cont(A_init,A,A_n,lambda_n):
    """Compute the new lambda using forward eular"""

    dA = A-A_n
    #print((beta1, lambda_n, A, A_T))
    #print (dt*((beta1*lambda_n*(A-A_init+dA/dt))/(A_init*(lambda_n+beta1))-beta2*lambda_n))
    return dt*((beta1*(A-A_T+dA/dt)))+lambda_n
    #return dt*((beta1*lambda_n*(A-A_T+dA/dt))/(A_T*(lambda_n+beta1))-beta2*lambda_n)+lambda_n

from datetime import datetime

def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(msh, default_real_type(1.0)) * dx)), op=MPI.SUM
    )
    return (1 / vol) * msh.comm.allreduce(fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)

def compute_root(f, Df, x0, y0, tol=1e-6, max_iter=100):
    """
    Computes the roots (lambdaA and lambdaV) of a system using Newton's method.

    Parameters:
    f : function
        The function representing the system.
    Df : function
        The Jacobian of the function.
    x0, y0 : float
        Initial approximations of the roots.
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations.

    Returns:
    (lambdaA, lambdaV) : tuple of floats
        Approximations of the roots of the system.
    """
    
    x, y = x0, y0
    for _ in range(max_iter):
        F = np.array(f(x, y))
        J = np.array(Df(x, y))
        
        if np.linalg.norm(F, ord=2) < tol:
            break
        
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("Jacobian is singular. Method failed.")
            return None
        
        x += delta[0]
        y += delta[1]
        
        if np.linalg.norm(delta, ord=2) < tol:
            break
    
    return x, y

def write_simulation_log(save_path, stop_criteria, final_area, final_lambda, simulation_time):
    """
    Writes a simulation log file with details about the run.

    Parameters:
        save_path (str): The directory to save the log file.
        stop_criteria (str): The stopping criteria of the simulation.
        final_area (float): The final area calculated in the simulation.
        final_lambda (float): The final lambda value from the simulation.
        simulation_time (float): The time taken for the simulation in seconds.
    """
    
    # Log file path
    log_file_path = os.path.join(save_path, 'simulation_log.txt')
    
    # Current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Content of the log
    log_content = f"""
    Simulation Log
    ==============
    Timestamp       : {timestamp}
    Time of Simulation (s) : {simulation_time:.2f}
    Stop Criteria   : {stop_criteria}
    Final Area      : {final_area:.4f}
    Final Lambda    : {final_lambda:.4f}
    ==============================
    """
    
    # Write or overwrite the log file
    with open(log_file_path, 'w') as file:
        file.write(log_content)
    
    print(f"Log file successfully written to: {log_file_path}")

def mpi_print(s):
    print(f"Rank {MPI.COMM_WORLD.rank}: {s}")


start = time.perf_counter()

gmsh.initialize()

h = 1e-1
R = 1
dt = 1e-3

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.rank

if mesh_comm.rank == gmsh_model_rank:

    membrane = gmsh.model.occ.addDisk(0, 0, 0, R, R) # x, y, z, x-radius, y-radius
    gmsh.model.occ.synchronize()

    # Make membrane a physical surface for GMSH to recognise when generating the mesh
    gdim = 1 # 2D Geometric Dimension of the Mesh
    gmsh.model.addPhysicalGroup(gdim, [membrane], 0) # Dimension, Entity tag, Physical tag

    # Generate 2D Mesh with uniform mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)

mesh, cell_markers, _ = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
mesh.name = "initial_mesh"
gmsh.finalize()


with XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_markers, mesh.geometry)

# Finalize GMSH

mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

gmsh.finalize()

P1 = element("Lagrange", mesh.basix_cell(), 2, gdim=2)
P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), gdim=2)

ME = functionspace(mesh, mixed_element([P1, P1], gdim=2))
MEx = functionspace(mesh, mixed_element([P2, P1], gdim=2))

V = functionspace(mesh, P2)
Q = functionspace(mesh, P1)

(Xh, Hh) = ufl.TrialFunctions(MEx)
(Xi, phi) = ufl.TestFunctions(MEx)

dof_ordered = order_dofs(mesh.geometry.dofmap)
area_init = area (dof_ordered,mesh.geometry.x[:,:2])

print(f"initial area {area_init}")      
q, v = ufl.TestFunctions(ME)

# +
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

Xh0 = Function(V) 
Xn = Function(V) 
Hh0 = Function(Q) 
Hn = Function(Q) 


# Split mixed functions
u1, u2 = ufl.split(u)
u10, u20 = ufl.split(u0)


au = 0.9
bu = 1.1
av = 0.8
bv = 0.95

# Interpolate initial condition

u.sub(0).interpolate(lambda x: (bu-au)*np.random.rand(x.shape[1]) +au)
u.sub(1).interpolate(lambda x: (bv-av)*np.random.rand(x.shape[1]) +av)

# Finalize initialization of solution variables
u.x.scatter_forward()

# Compute oriented normal vectors
n = ufl.CellNormal(mesh)
xc = ufl.SpatialCoordinate(mesh)
r = xc / ufl.sqrt(ufl.dot(xc, xc))  # Radial unit vector
sign = ufl.sign(ufl.dot(n, r))  # Adjust sign for orientation consistency
n_oriented = sign * n  # Corrected normal vector

# Define coordinate expression for interpolation
def x_exp(x):
    """Expression for coordinate interpolation."""
    return np.vstack((x[0], x[1]))

# Interpolate mean curvature field
H_expr = Expression(div(sign * n), Q.element.interpolation_points())
Hh0.interpolate(H_expr)

# Define geometric function space and functions
V_geom = functionspace(mesh, P2)
X_geom = Function(V_geom)

# Define tangential projection operator
def T(Xn, X, n):
    """Compute tangential projection operator."""
    return (X - Xn) / dt - inner((X - Xn) / dt, n) * n

# Initialize displacement field
Xh0.interpolate(x_exp)
X_geom.interpolate(Xh0)


# Define species vector
u_vec = ufl.as_vector([u10, u20])

# Define simulation parameters
kp = 1.5  # Reaction coefficient
k = dt  # Time step size
d = 10  # Diffusion coefficient
gamma = 100  # Reaction rate constant
a, b = 0.1, 0.9  # Model parameters
tol = 1e-4  # Tolerance for solver

d1, d2 = 1.0, d  # Diffusion coefficients

# Lagrange multipliers and area constraints
beta1, beta2 = -10, 0.1
A_T = area_init  # Target area
lambda_n = 20  # Initial Lagrange multiplier

# Stiffness coefficients
ks, kb = 1, 1e-2

# Define Lagrange multiplier as a constant field
lamb = Constant(mesh, PETSc.ScalarType(lambda_n))

# Tangential projection term
T = (Xn, Xh0, n_oriented)

# Define weak formulation of the system
F = (
    ((u1 - u10) / k) * q * dx
    - div(inner(u1 * T)) * q * dx
    + d1 * inner(grad(u1), grad(q)) * dx
    - (gamma * (u1**2 * u2 - u1 + a)) * q * dx
    + ((u2 - u20) / k) * v * dx
    - div(inner(u2 * T)) * v * dx
    + d2 * inner(grad(u2), grad(v)) * dx
    - (gamma * (-u1**2 * u2 + b)) * v * dx
)

# Create nonlinear problem and Newton solver
lamb = Constant(mesh, PETSc.ScalarType(lambda_n))
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# Customize the linear solver
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

# Time stepping initialization
t = 0.0

# Output file setup
mesh.name = f"mesh_at_t{t}"

# Reduce run time if running on a CI server
T = 3 * dt if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ else 20000 * dt

# Define save path for results
save_path = fr"Schnakenberg2O/test/kp{kp}"
os.makedirs(save_path, exist_ok=True)

# Extract function subspaces
V0, dofs = ME.sub(0).collapse()
u1, u2 = u.sub(0), u.sub(1)

# Interpolate initial geometry
V_geom = functionspace(mesh, P2)
X_geom = Function(V_geom)
np.testing.assert_allclose(mesh.geometry.x, V_geom.tabulate_dof_coordinates(), atol=1e-13)
X_geom.interpolate(Xh0)
mesh.geometry.x[:, :2] = X_geom.x.array.reshape(-1, mesh.geometry.dim)

# Setup output files
file1 = XDMFFile(MPI.COMM_WORLD, os.path.join(save_path, "outputu1.xdmf"), "w")
file1.write_mesh(mesh)
file1.write_function(u1, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")

file2 = XDMFFile(MPI.COMM_WORLD, os.path.join(save_path, "outputH_.xdmf"), "w")
file2.write_mesh(mesh)
file2.write_function(Hh0, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")

# Initialize data storage files
fileA = open(os.path.join(save_path, 'Area.dat'), 'w')
filelamb = open(os.path.join(save_path, 'Lamb.dat'), 'w')
fileL2 = open('L2.dat', 'w')

# Simulation control variables
norm_stop = 0.1e-6
z, inc, impresion = 0, 0, 100

# Initialize solution arrays
u0.x.array[:] = u.x.array
A_init = area(dof_ordered, mesh.geometry.x[:, :2])
A_n = area_init  # Initial area assumption
X = Function(MEx)

# Data tracking lists
t_array, A_array, lamb_array, H_array = [], [], [], []

t_array.append(t)
A_array.append(A_n)
lamb_array.append(lambda_n)
H_array.append(np.max(Hh0.x.array[:]))

# Write initial values to file
fileA.write(f"{t}\t{A_n}\n")
filelamb.write(f"{t}\t{lambda_n}\n")

# Define stopping criteria
stop_criteria = "successful simulation"

# Time-stepping loop
while t < T:
    t += dt
    lamb = Constant(mesh, PETSc.ScalarType(lambda_n))
    
    # Define bilinear and linear forms
    ax1 = inner((Xh / k), phi * n_oriented) * dx + ks * inner(Hh, phi) * dx
    ax2 = inner(Hh * n_oriented, Xi) * dx - inner(grad(Xh), grad(Xi)) * dx
    Lx1 = dot((Xh0 / k), phi * n_oriented) * dx + inner(kp * u10, phi) * dx
    Lx2 = 0
    
    ax = ax1 + ax2
    Lx = Lx1
    
    # Assemble linear system
    bilinear_form = form(ax)
    linear_form = form(Lx)
    Ax = assemble_matrix(bilinear_form)
    Ax.assemble()
    
    b = create_vector(linear_form)
    assemble_vector(b, linear_form)
    
    # Solve linear system
    solverx = PETSc.KSP().create(mesh.comm)
    solverx.setOperators(Ax)
    solverx.setType(PETSc.KSP.Type.PREONLY)
    solverx.getPC().setType(PETSc.PC.Type.LU)
    solverx.solve(b, X.vector)
    
    Xn, Hn = X.sub(0).collapse(), X.sub(1).collapse()
    np.testing.assert_allclose(mesh.geometry.x, V_geom.tabulate_dof_coordinates(), atol=1e-13)
    X_geom.interpolate(X.sub(0))
    Hh0.interpolate(X.sub(1))
    mesh.geometry.x[:, :2] = X_geom.x.array.reshape(-1, mesh.geometry.dim)
    Xh0.x.array[:] = X_geom.x.array[:]
    
    # Update simulation parameters
    A = area(dof_ordered, mesh.geometry.x[:, :2])
    lang_mult = lambda_cont(A_init, A, A_n, lambda_n)
    r = solver.solve(u)
    inc += 1
    l2_norm = np.linalg.norm(u.x.array[dofs] - u0.x.array[dofs]) / dt
    
    # Update solution arrays
    u0.x.array[:] = u.x.array
    lambda_n = lang_mult
    A_n = A
    t_array.append(t)
    A_array.append(A_n)
    lamb_array.append(lambda_n)
    H_array.append(np.max(Hh0.x.array[:]))
    
    # Write data to files
    fileL2.write(f"{t}\t{l2_norm}\n")
    fileA.write(f"{t}\t{A_n}\n")
    filelamb.write(f"{t}\t{lambda_n}\n")
    
    # Break conditions
    if A_n < 1e-6 or A_n > 1e2:
        print("=" * 50)
        print(f"🔴 Breaking loop: A_n ({A_n:.6f}) {'< 1e-6' if A_n < 1e-6 else '> 1e2'}")
        print(f"{'Step':<20} {int(t / dt)}, Iterations: {r[0]}")
        print(f"{'Area:':<20} {A:.6f}")
        print(f"{'Lagrange Multiplier:':<20} {lang_mult:.6f}")
        print("=" * 50)
        stop_criteria = f"A_n ({A_n}) {'< 1e-6' if A_n < 1e-6 else '> 1e2'}"
        break
    
    # Save intermediate results
    if int(inc / impresion - 1) == z:
        mesh.name = f"mesh_at_t{t}"
        file1.write_mesh(mesh)
        file1.write_function(u1, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
        file2.write_mesh(mesh)
        file2.write_function(Hh0, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
        z += 1
        
        print("=" * 40)
        print(f"{'Step':<20} {int(t / dt)}, Iterations: {r[0]}")
        print(f"{'Area:':<20} {A:.6f}")
        print(f"{'Lagrange Multiplier:':<20} {lang_mult:.6f}")
        print("=" * 40)

# Close output files
file1.close()
fileL2.close()
fileA.close()
filelamb.close()

print(f"Final L2 norm: {l2_norm:.6e}")

# Configure matplotlib settings
plt.rc('text', usetex=False)  # Disable native LaTeX rendering
plt.rc('font', family='serif')

# Plot Area vs Time
plt.figure()
plt.plot(t_array, A_array, linewidth=2.5, label='Area')
plt.axhline(y=A_T, color='red', linestyle='--', linewidth=2, label='$A_T$')  # Target area line
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # x-axis
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # y-axis
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$A$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'M2Area{kp}.png'))
plt.close()

# Plot Lambda vs Time
plt.figure()
plt.plot(t_array, lamb_array, linewidth=2.5, label='$\lambda$', color='orange')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$\lambda$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'M2lambda{kp}.png'))
plt.close()

# Plot Maximum H vs Time
plt.figure()
plt.plot(t_array, H_array, linewidth=2.5, label='$H$', color='orange')
plt.xlabel('$t$', fontsize=16)
plt.ylabel('$H$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'M2H_max{kp}.png'))
plt.close()

# Log execution time
end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} seconds")

# Write simulation log
write_simulation_log(save_path, stop_criteria, A, lambda_n, end - start)
