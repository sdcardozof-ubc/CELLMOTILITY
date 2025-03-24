"""
This script performs ESFEM simulation under different conditions.

Author: Sergio Daniel Cardozo
Date: March 11, 2025
Version: 7.0

Description:
- Compute the mesh coordinates evolving over time forced by different factos, mainly mean curvcature.
- Outputs xdmf file readable in praview, an plots of area, curvature and lambda.

Usage:
Run this script using Python 3.9 or later:
    python3 test_movemesh.py
"""

import os
import time
import numpy as np
import math as math

os.environ["FFCX_JIT_TIMEOUT"] = "300"

# MPI and PETSc imports
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
from dolfinx import mesh  # type: ignore
from dolfinx.fem import (  #type:ignore
    Function, functionspace, Expression, form, Constant  # type: ignore
)
from dolfinx.fem.petsc import ( #type:ignore
    assemble_matrix, assemble_vector, create_vector  # type: ignore
)
from dolfinx.io import XDMFFile, gmshio  # type: ignore
from ufl import dx, grad, inner, div, dot, pi, sin, exp, cos  # type: ignore

import os
import time
import numpy as np
from datetime import datetime

# MPI and PETSc imports
from mpi4py import MPI  # type: ignore
from petsc4py import PETSc  # type: ignore

# Import GMSH
try:
    import gmsh  # type: ignore
except ImportError:
    import sys
    print("This program requires gmsh to be installed")
    sys.exit(0)

# Mesh and Geometry Parameters
h = 3e-1
R = 1
dt = 1e-3
A_T = np.pi  # Target area

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx mesh from a Gmsh model and save to file."""
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0, gdim=2)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    
    with XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(0, 1)
        file.write_mesh(msh)

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

def area(dofs, coordinates):
    """Compute the area using DOFs and coordinates."""
    num_ele, _ = dofs.shape
    area = sum(
        coordinates[dofs[i, 0], 0] * coordinates[dofs[i, 2], 1] -
        coordinates[dofs[i, 0], 1] * coordinates[dofs[i, 2], 0] +
        coordinates[dofs[i, 2], 0] * coordinates[dofs[i, 1], 1] -
        coordinates[dofs[i, 2], 1] * coordinates[dofs[i, 1], 0]
        for i in range(num_ele)
    )
    return abs(area) / 2

def lambda_cont(A_init, A, A_n, lambda_n,lambda_0):
    """Compute new lambda using the forward Euler method.
    
    Returns:
        lambda_new: Updated lambda value.
        term1: (beta1 * lambda_n * (A - A_T + dA / dt)) / (A_T * (lambda_n + beta1))
        term2: beta2 * lambda_n
    """
    dA = A - A_n
    num = (beta1 * lambda_n * (A - A_T + dA / dt));
    den = (A_T * (lambda_n + lambda_0));
    term1 = (beta1 * lambda_n * (A - A_T + dA / dt))/ (A_T * (lambda_n + lambda_0));
    term2 = beta2 * lambda_n
    lambda_new = dt * (term1 - term2) + lambda_n
    
    return lambda_new, term1, term2, num, den

def write_simulation_log(save_path, stop_criteria, final_area, final_lambda, simulation_time):
    """Writes a simulation log file with details about the run."""
    log_file_path = os.path.join(save_path, 'simulation_log.txt')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_content = f"""
    Simulation Log
    ==============
    Timestamp       : {timestamp}
    Simulation Time : {simulation_time:.2f} s
    Stop Criteria   : {stop_criteria}
    Final Area      : {final_area:.4f}
    Final Lambda    : {final_lambda:.4f}
    ==============================
    """
    with open(log_file_path, 'w') as file:
        file.write(log_content)
    print(f"Log file successfully written to: {log_file_path}")

def mpi_print(s):
    """Print message with MPI rank."""
    print(f"Rank {MPI.COMM_WORLD.rank}: {s}")

# Initialize GMSH
start = time.perf_counter()
gmsh.initialize()

# Set up parallel processing
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.rank

if mesh_comm.rank == gmsh_model_rank:
    membrane = gmsh.model.occ.addDisk(0, 0, 0, R, R)
    gmsh.model.occ.synchronize()
    gdim = 1  # 2D geometric dimension
    gmsh.model.addPhysicalGroup(gdim, [membrane], 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)

# Create mesh from GMSH model
mesh, cell_markers, _ = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
mesh.name = "initial_mesh"


# Save mesh
with XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_markers, mesh.geometry)

# Finalize GMSH
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
gmsh.finalize()

# Define function spaces
P1 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), gdim=2)
P2 = element("Lagrange", mesh.basix_cell(), 2, gdim=2)
MEx = functionspace(mesh, mixed_element([P1, P2], gdim=2))

# Compute normal vectors
n = ufl.CellNormal(mesh)
xc = ufl.SpatialCoordinate(mesh)
r = xc / ufl.sqrt(ufl.dot(xc, xc))
sign = ufl.sign(ufl.dot(n, r))
n_oriented = sign * n

# Define scalar and vector function spaces
V = functionspace(mesh, P1)  # Displacement field
Q = functionspace(mesh, P2)  # Curvature field

# Define variational problem
(Xh, Hh) = ufl.TrialFunctions(MEx)
(Xi, phi) = ufl.TestFunctions(MEx)

# Initialize functions
Xh0 = Function(V) 
Xn = Function(V) 
Hh0 = Function(Q) 
Hn = Function(Q) 
H_expr = Expression(div(sign * n), Q.element.interpolation_points())

# Define geometric function spaces
V_geom = functionspace(mesh, P1)
Q_geom = functionspace(mesh, P2)
u_geom = Function(V_geom)
X_geom = Function(V_geom)
H_geom = Function(Q_geom)

# Coordinate interpolation expressions
def x_exp(x):
    """Expression for coordinates."""
    return np.vstack((x[0], x[1]))

def U_exp(x):
    """Expression for velocity."""
    return np.vstack((x[0], x[1]))

# Gather DOF map and coordinates from all processes
dofs = MPI.COMM_WORLD.gather(mesh.geometry.dofmap, root=0)
coordinates = MPI.COMM_WORLD.gather(mesh.geometry.x, root=0)

if rank == 0:
    dofs = np.concatenate(dofs)
    coordinates = np.concatenate(coordinates)
    dof_ordered = order_dofs(dofs)  # Order DOFs once
    area_init = area(dof_ordered, coordinates)  # Compute initial area once
else:
    area_init = None

# Broadcast computed area_init to all processes
area_init = MPI.COMM_WORLD.bcast(area_init, root=0)
print(f"Initial area: {area_init}")

# Interpolate initial conditions
Xh0.interpolate(x_exp)
Hh0.interpolate(H_expr)
X_geom.interpolate(Xh0)
H_geom.interpolate(H_expr)
x = ufl.SpatialCoordinate(mesh)

# Define simulation parameters
k = dt  # Time step size
ks = 1e-1  # Surface tension coefficient
kb = 1e-2  # Bending rigidity
kp = 1.5e-1  # Curvature penalty coefficient

# Lagrange multipliers
beta1 = 2
beta2 = 10
lambda_n = 2  # Initial lambda
lambda_0 = 2  # Initial lambda

# Compute angular coordinate


# Define function for displacement field
u = Function(Q)

# Initialize time-stepping variables
t = 0.0
mesh.name = f"mesh_at_t{t}"

# Reduce runtime if running on CI server
T = 3 * dt if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ else 20000 * dt

# Define save path
save_path = fr"MeanCurvature2O/Results/A0_{area_init:.3f}"
os.makedirs(save_path, exist_ok=True)

# Setup output file for function u
file1 = XDMFFile(MPI.COMM_SELF, os.path.join(save_path, f"outputU_{MPI.COMM_WORLD.rank}.xdmf"), "w")
file1.write_mesh(mesh)
file1.write_function(u, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")

# Initialize data storage files
fileA = open(os.path.join(save_path, 'Area.dat'), 'w')
filelamb = open(os.path.join(save_path, 'Lamb.dat'), 'w')

# Simulation control variables
z, inc, impresion = 0, 0, 50

# Initialize geometric function
X = Function(MEx)

# Compute initial area
A_init = area(dof_ordered, mesh.geometry.x[:, :2])
A_n = A_init

# Data tracking lists
t_array, A_array, lamb_array, H_array = [], [], [], []

term1_,term2_,num_, den_ = [], [], [],[]

t_array.append(t)
A_array.append(A_n)
lamb_array.append(lambda_n)
H_array.append(np.max(H_geom.x.array[:]))

# Write initial values to file
fileA.write(f"{t}\t{A_n}\n")
filelamb.write(f"{t}\t{lambda_n}\n")

# Define stopping criteria
stop_criteria = "successful simulation"

# # Additional parameters for u expression
# a0 = Constant(mesh, PETSc.ScalarType(1))
# ohm = Constant(mesh, PETSc.ScalarType(0.2))
# d0 = Constant(mesh, PETSc.ScalarType(0.1))

# Update time-dependent parameters
# a = a0 * (1 + 0.5 * np.cos(ohm * t))
# d = d0 + 1.2*sin(ohm * t)


# class u_exp:
#     def __init__(self, t):
#         self.t = t
#         self.a0 = 1.0  # Initial amplitude
#         self.sigma = 0.5  # Width of the Gaussian
#         self.v = 0.1  # Speed of movement in theta

#     def __call__(self, x):
#         theta = np.arctan2(x[1], x[0])  # Compute theta
#         theta_t = 0.3 + self.v * self.t  # Linear shift in theta over time
#         a_t = self.a0 * np.exp(-((self.t - 0.5) ** 2) / (2 * self.sigma ** 2))  # Time-evolving Gaussian

#         return a_t * np.exp(-((theta - theta_t) ** 2) / (2 * self.sigma ** 2))


# Update time-dependent parameters
a0 = 1
ohm = 0.5
d0 = 0.2

# Compute angular coordinate
theta = ufl.atan2(x[1], x[0])

a = a0
d = d0
u_exp = Expression(1 + a * exp(-2 * (theta - 2) ** 2) * sin(1.2 * theta + d), Q.element.interpolation_points())
u.interpolate(u_exp)

# Time-stepping loop
while t < T:
    t += dt
    inc += 1

    # ut = u_exp(t)
    # u.interpolate(ut)

    # Update Lagrange multiplier
    lamb = Constant(mesh, PETSc.ScalarType(lambda_n))

    # Define bilinear and linear forms
    ax1 = inner((Xh / k), phi * n_oriented) * dx + ks * inner(Hh, phi) * dx
    ax2 = inner(Hh * n_oriented, Xi) * dx - inner(grad(Xh), grad(Xi)) * dx
    Lx1 = dot((Xh0 / k), phi * n_oriented) * dx + kp * u * phi * dx - lamb * phi * dx
    
    ax = ax1 + ax2
    Lx = Lx1

    # Assemble and solve linear system
    bilinear_form = form(ax)
    linear_form = form(Lx)
    Ax = assemble_matrix(bilinear_form)
    Ax.assemble()
    b = create_vector(linear_form)
    assemble_vector(b, linear_form)

    solverx = PETSc.KSP().create(mesh.comm)
    solverx.setOperators(Ax)
    solverx.setType(PETSc.KSP.Type.PREONLY)
    solverx.getPC().setType(PETSc.PC.Type.LU)
    solverx.solve(b, X.vector)

    # Split and update solution
    Xn, Hn = X.sub(0).collapse(), X.sub(1).collapse()
    np.testing.assert_allclose(mesh.geometry.x, V_geom.tabulate_dof_coordinates(), atol=1e-13)
    
    # Update mesh geometry
    X_geom.interpolate(X.sub(0))
    Hh0.interpolate(X.sub(1))
    mesh.geometry.x[:, :2] = X_geom.x.array.reshape(-1, mesh.geometry.dim)
    Xh0.x.array[:] = X_geom.x.array[:]
    
    # Compute new area and lambda multiplier
    A = area(dof_ordered, mesh.geometry.x[:, :2])
    lang_mult, term1, term2,num,den = lambda_cont(A_init, A, A_n, lambda_n,lambda_0)
    l2_norm = np.linalg.norm(A)
    #print(f"Step {int(t / dt)}: A = {l2_norm}")

    # Update variables
    lambda_n = lang_mult
    A_n = A
    
    # Store results
    num_.append(num)
    den_.append(den)
    term1_.append(term1)
    term2_.append(term2)
    t_array.append(t)
    A_array.append(A_n)
    lamb_array.append(lambda_n)
    H_array.append(np.max(Hh0.x.array[:]))
    
    # Write data to files
    fileA.write(f"{t}\t{A_n}\n")
    filelamb.write(f"{t}\t{lambda_n}\n")

    # Break conditions
    if A_n < 1e-6:
        print(f"Breaking loop: A_n ({A_n}) < 1e-6")
        stop_criteria = "A_n ({A_n}) < 1e-6"
        break
    if A_n > 1e2:
        print(f"Breaking loop: A_n ({A_n}) > 1e2")
        stop_criteria = "A_n ({A_n}) > 1e2"
        break
    
    # Save intermediate results
    if int(inc / impresion - 1) == z:
        mesh.name = f"mesh_at_t{t}"
        file1.write_mesh(mesh)
        file1.write_function(u, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
        z += 1

        print("=" * 40)
        print(f"{'Step':<20} {int(t / dt)}, Iterations: NA")
        print(f"{'Area:':<20} {A:.6f}")
        print(f"{'Lagrange Multiplier:':<20} {lang_mult:.6f}")
        print("=" * 40)

# Close output files
file1.close()
fileA.close()
filelamb.close()


# Configure matplotlib settings
plt.rc('text', usetex=False)  # Disable LaTeX rendering
plt.rc('font', family='serif')

# Plot Area vs Time
plt.figure()
plt.plot(t_array, A_array, linewidth=2.5, label='Area')
plt.axhline(y=A_T, color='red', linestyle='--', linewidth=2, label=r'$A_T$')  # Target area line
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # x-axis
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # y-axis
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$A$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'Area_{A_T:.3f}.png'))
plt.close()

# Plot Lambda vs Time
plt.figure()
plt.plot(t_array, lamb_array, linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$\lambda$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'lambda_{A_T:.3f}.png'))
plt.close()

# Plot term1 vs Time
plt.figure()
plt.plot(t_array[1:], term1_, linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_1$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'term1_{A_T:.3f}.png'))
plt.close()

# Plot term2 vs Time
plt.figure()
plt.plot(t_array[1:], term2_, linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_2$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'term2_{A_T:.3f}.png'))
plt.close()

# Plot num vs Time
plt.figure()
plt.plot(t_array[1:], num_, linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_0$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'num_{A_T:.3f}.png'))
plt.close()

# Plot den vs Time
plt.figure()
plt.plot(t_array[1:], den_, linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_0$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'den_{A_T:.3f}.png'))
plt.close()

# Plot term1 vs Time (semilogy)
plt.figure()
plt.semilogy(t_array[1:], np.abs(term1_), linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_1$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'term1y_{A_T:.3f}.png'))
plt.close()

# Plot term2 vs Time (semilogy)
plt.figure()
plt.semilogy(t_array[1:], np.abs(term2_), linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_2$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'term2y_{A_T:.3f}.png'))
plt.close()

# Plot num vs Time (semilogy)
plt.figure()
plt.semilogy(t_array[1:], np.abs(num_), linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_0$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'numy_{A_T:.3f}.png'))
plt.close()

# Plot den vs Time (semilogy)
plt.figure()
plt.semilogy(t_array[1:], np.abs(den_), linewidth=2.5, label=r'$\lambda$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$term_0$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'deny_{A_T:.3f}.png'))
plt.close()

# Plot Maximum H vs Time
plt.figure()
plt.plot(t_array, H_array, linewidth=2.5, label='$H$', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$H$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(os.path.join(save_path, f'H_max_{A_T:.3f}.png'))
plt.close()

# Log execution time
end = time.perf_counter()
print(f"Elapsed time: {end - start:.4f} seconds")

# Write simulation log
write_simulation_log(save_path, stop_criteria, A, lambda_n, end - start)
