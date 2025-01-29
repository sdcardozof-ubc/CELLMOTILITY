"""
This script performs ESFEM simulation under different conditions.

Author: Sergio Daniel Cardozo
Date: January 9, 2025
Version: 6.0

Description:
- Compute the mesh coordinates evolving over time forced by different factos, mainly mean curvcature.
- Outputs xdmf file readable in praview, an plots of area, curvature and lambda.

Usage:
Run this script using Python 3.9 or later:
    python3 test_movemesh.py
"""

import os

from mpi4py import MPI # type: ignore
from petsc4py import PETSc # type: ignore
import matplotlib as mpl # type: ignore
mpl.use('Agg')
import matplotlib.pyplot as plt # type: ignore

import basix.ufl # type: ignore
import dolfinx # type: ignore
from basix.ufl import element, mixed_element # type: ignore
from dolfinx import mesh # type: ignore
from dolfinx.fem import Function, functionspace, Expression, form, Constant # type: ignore
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector # type: ignore
from dolfinx.io import XDMFFile, gmshio # type: ignore
from ufl import dx, grad, inner, div,dot # type: ignore
import ufl # type: ignore
import numpy as np
import time 

try:
    import gmsh  # type: ignore
except ImportError:
    import sys
    print("This program requires gmsh to be installed")
    sys.exit(0)

def gmsh_circle(model: gmsh.model, name: str) -> gmsh.model:
    """Create a Gmsh model of a circle.
    """

    model.add(name)
    model.setCurrent(name)
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", h)

    circle = model.occ.addDisk(0, 0, 0, R, R, tag=1)

    # Synchronize OpenCascade representation with gmsh model
    model.occ.synchronize()

    # Add physical marker for cells. It is important to call this
    # function after OpenCascade synchronization
    model.add_physical_group(dim=1, tags=[circle])

    # Generate the mesh
    model.mesh.generate(dim=1)
    model.mesh.setOrder(2)

    return model

def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file.
    """
    msh, ct, ft = gmshio.model_to_mesh(model, comm, rank=0, gdim=2)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    
    with XDMFFile(msh.comm, filename, mode) as file:
        msh.topology.create_connectivity(0, 1)
        file.write_mesh(msh)
        # file.write_meshtags(ct, msh.geometry, geometry_xpath=f"/Xdmf/mesh/Grid[@Name='{msh.name}']/Geometry")
        # file.write_meshtags(ft, msh.geometry, geometry_xpath=f"/Xdmf/mesh/Grid[@Name='{msh.name}']/Geometry")

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

# Create a mesh from the GMSH model

mesh, cell_markers, _ = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
mesh.name = "initial_mesh"
gmsh.finalize()

with XDMFFile(MPI.COMM_WORLD, "circle.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(cell_markers, mesh.geometry)

# Finalize GMSH

# gmsh.initialize()
# gmsh.option.setNumber("General.Terminal", 0)

# dt = 1e-3
# h = 1e-2
# R = 1


# # Create model
# model = gmsh.model()
# model = gmsh_circle(model, "circle")
# model.setCurrent("circle")

# create_mesh(MPI.COMM_SELF, model, "circle", f"out_gmsh/mesh_rankCircle_{MPI.COMM_WORLD.rank}.xdmf", "w")

# with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh_rankCircle_0.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh(name="circle")

# mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)

# gmsh.finalize

P1 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,), gdim=2)
P2 = element("Lagrange", mesh.basix_cell(), 2, gdim=2)

MEx = functionspace(mesh, mixed_element([P1, P2], gdim=2))

n = ufl.CellNormal(mesh)
xc = ufl.SpatialCoordinate(mesh)
r = xc/ufl.sqrt(ufl.dot(xc, xc))
sign = ufl.sign(ufl.dot(n, r))
n_oriented = sign*n

V = functionspace(mesh, P1)
Q = functionspace(mesh, P2)

# Define variational problem
(Xh, Hh) = ufl.TrialFunctions(MEx)
(Xi, phi) = ufl.TestFunctions(MEx)


Xh0 = Function(V) 
Xn = Function(V) 
Hh0 = Function(Q) 
Hn = Function(Q) 

H_expr = Expression(div(sign*n), Q.element.interpolation_points())

# Permute dofmap
V_geom = dolfinx.fem.functionspace(mesh, P1)
Q_geom = dolfinx.fem.functionspace(mesh, P2)
u_geom = dolfinx.fem.Function(V_geom)
X_geom = dolfinx.fem.Function(V_geom)
H_geom = dolfinx.fem.Function(Q_geom)

# Xinit = dolfinx.fem.Function(V)
# Xinit.interpolate(lambda x: (x[0] * (1 + 0.3*np.sin(2*np.pi*(x[0]+x[1]))),x[1]* (1 + 0.3*np.sin(2*np.pi*(x[0]+x[1])))))


# u_geom = dolfinx.fem.Function(V_geom)
# np.testing.assert_allclose(mesh.geometry.x, V_geom.tabulate_dof_coordinates(), atol=1e-13)
# u_geom.interpolate(Xinit)

# mesh.geometry.x[:, :2] = u_geom.x.array.reshape(-1, mesh.geometry.dim)


def x_exp(x):
    """Expression for coordinates"""
    return np.vstack((x[0],x[1]))


dofs = MPI.COMM_WORLD.gather(mesh.geometry.dofmap, root=0)
coordinates = MPI.COMM_WORLD.gather(mesh.geometry.x, root=0)

print(f"len{type(mesh.geometry.x)}")
print(f"len{type(coordinates)}")

if rank == 0:
    dofs = np.concatenate(dofs)
    coordinates = np.concatenate(coordinates)

    dof_ordered = order_dofs(dofs)  # Order dofs only once
    area_init = area(dof_ordered, coordinates)   # Calculate area_init only once
else:
    area_init = None

area_init = MPI.COMM_WORLD.bcast(area_init, root=0)
print(F"area= {area_init}")
# Broadcast the computed area_init from rank 0 to all other processes

# #making the area unitary
# k = np.sqrt(np.pi/area_init)
# mesh.geometry.x[:, :2] = k * mesh.geometry.x[:, :2]

Xh0.interpolate(x_exp)
Hh0.interpolate(H_expr)

X_geom.interpolate(Xh0)

H_geom.interpolate(H_expr)

k = dt
ks = 1
kb = 1e-2

beta1 = -1
beta2 = 0.1
A_T = area_init
lambda_n = 0

lamb = Constant(mesh, PETSc.ScalarType(lambda_n))

ax1 = inner((Xh / k),phi*n_oriented)*dx \
    + ks*inner(Hh,phi)*dx\
    # + kb*inner(grad(Hh),grad(phi))*dx - 0.5* kb*Hh0**2*Hh*phi*dx\
    
ax2 = inner(Hh*n_oriented,Xi)*dx - inner(grad(Xh),grad(Xi))*dx

Lx1 = dot((Xh0 / k),phi*n_oriented)*dx + lamb*phi*dx
Lx2 = 0

ax = ax1 + ax2
Lx = Lx1

bilinear_form = form(ax)
linear_form = form(Lx)

# Step in time
t = 0.0

# Output file
name = "mesh_at_t"+str(t)
mesh.name = name

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 5000 * dt

save_path = fr'MeanCurvature2O/1Proof/b1{beta1}'

os.makedirs(save_path, exist_ok=True)

file1 = XDMFFile(MPI.COMM_SELF, os.path.join(save_path,f"outputH__{MPI.COMM_WORLD.rank}.xdmf"), "w")
file1.write_mesh(mesh)
file1.write_function(Hh0, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")


# # file1.write_function([Xh0], t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")
# file1 = MPI.COMM_WORLD.bcast(file1, root=0)

fileA = open( os.path.join(save_path,'Area.dat'), 'w') 
filelamb = open( os.path.join(save_path,'Lamb.dat'), 'w') 

z = 0
inc = 0 
impresion = 50

X = Function(MEx)

A_init = area (dof_ordered,mesh.geometry.x[:,:2])
A_n = A_init  # esto puede estar mal

t_array = []
A_array = []
lamb_array = []
H_array = []

fileA.write(f"{t}\t{A_n}\n")
filelamb.write(f"{t}\t{lambda_n}\n")

t_array.append(t)  # Llenamos el primer array con los valores de X
A_array.append(A_n)  # Llenamos el segundo array con algún cálculo, en este caso X al cuadrado
lamb_array.append(lambda_n)
H_array.append(np.max(H_geom.x.array[:])) 

stop_criteria = "successfull simulation"

while (t < T):
    t += dt
    inc += 1

    lamb = Constant(mesh, PETSc.ScalarType(lambda_n))

    ax1 = inner((Xh / k),phi*n_oriented)*dx \
    + ks*inner(Hh,phi)*dx\
    # + kb*inner(grad(Hh),grad(phi))*dx - 0.5* kb*Hh0**2*Hh*phi*dx\
    
    ax2 = inner(Hh*n_oriented,Xi)*dx - inner(grad(Xh),grad(Xi))*dx

    Lx1 = dot((Xh0 / k),phi*n_oriented)*dx + lamb*phi*dx
    Lx2 = 0

    ax = ax1 + ax2
    Lx = Lx1

    bilinear_form = form(ax)
    linear_form = form(Lx)
    
    # Solve linear problem
    Ax = assemble_matrix(bilinear_form)
    Ax.assemble()

    b = create_vector(linear_form)
    assemble_vector(b, linear_form)

    solverx = PETSc.KSP().create(mesh.comm)
    solverx.setOperators(Ax)
    solverx.setType(PETSc.KSP.Type.PREONLY)
    solverx.getPC().setType(PETSc.PC.Type.LU)

    solverx.solve(b, X.vector)

    # Split the mixed solution and collapse
    Xn, Hn = X.sub(0).collapse(), X.sub(1).collapse()

    np.testing.assert_allclose(mesh.geometry.x, V_geom.tabulate_dof_coordinates(), atol=1e-13)

    num=inc/impresion-1

    # Move the mesh

    X_geom.interpolate(X.sub(0))
    Hh0.interpolate(X.sub(1))

    mesh.geometry.x[:,:2] = X_geom.x.array.reshape(-1, mesh.geometry.dim)
    Xh0.x.array[:] =  X_geom.x.array[:]
    #Hh0.x.array[:] =  H_geom.x.array[:]

    A = area (dof_ordered,mesh.geometry.x[:,:2])
    lang_mult = lambda_cont(A_init,A,A_n,lambda_n)
    l2_norm = np.linalg.norm(A)

    print(f"Step {int(t/dt)}: A = {l2_norm}")

    lambda_n = lang_mult
    A_n = A
    
    t_array.append(t)  # Llenamos el primer array con los valores de X
    A_array.append(A_n)  # Llenamos el segundo array con algún cálculo, en este caso X al cuadrado
    lamb_array.append(lambda_n) 
    H_array.append(np.max(Hh0.x.array[:])) 


    fileA.write(f"{t}\t{A_n}\n")
    filelamb.write(f"{t}\t{lambda_n}\n")

    # Break condition
    if A_n < 1e-6:
        print(f"Breaking loop: A_n ({A_n}) < 1e-6")
        stop_criteria = "A_n ({A_n}) < 1e-6"
        break

    if A_n > 1e2:
        print(f"Breaking loop: A_n ({A_n}) > 1e2")
        stop_criteria = "A_n ({A_n}) > 1e2"
        break
    
    if (int(num)==z) or (int(num-z)==0):

        # Save solution to file (VTK)

        name = "mesh_at_t"+str(t)
        mesh.name = name

        file1.write_mesh(mesh)
        file1.write_function(Hh0, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']")

        z += 1
    
file1.close()
fileA.close()
filelamb.close()

plt.rc('text', usetex=False)  # Desactivar LaTeX nativo, usar motor matemático de matplotlib
plt.rc('font', family='serif')

# Supongamos que ya tienes los arrays t_array, A_array y lamb_array definidos.

# Primer gráfico: X vs Y1 (t vs Area)
plt.figure()  # Crea una nueva figura
plt.plot(t_array, A_array, linewidth=2.5, label=r'Area')  # Se mantiene la misma etiqueta sin LaTeX 
plt.axhline(y=A_T, color='red', linestyle='--', linewidth=2, label=r'$A_T$')  # Add horizontal line
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # x-axis
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)  # y-axis
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$A$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)  # Aumentar el tamaño de los ticks
#plt.legend(fontsize=14)
plt.savefig(os.path.join(save_path, f'Area{ks}.png'))  # Guarda la figura como archivo PNG
plt.close()  # Cierra la figura para no mostrarla

# Segundo gráfico: X vs Y2 (t vs lambda)
plt.figure()  # Crea una nueva figura
plt.plot(t_array, lamb_array, linewidth=2.5, label=r'lambda', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$\lambda$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)  # Aumentar el tamaño de los ticks
#plt.legend(fontsize=14)
plt.savefig(os.path.join(save_path, f'lambda{ks}.png'))  # Guarda la figura como archivo PNG
plt.close()  # Cierra la figura para no 

# Segundo gráfico: X vs Y2 (t vs lambda)
plt.figure()  # Crea una nueva figura
plt.plot(t_array, H_array, linewidth=2.5, label=r'lambda', color='orange')
plt.xlabel(r'$t$', fontsize=16)
plt.ylabel(r'$H$', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)  # Aumentar el tamaño de los ticks
#plt.legend(fontsize=14)
plt.savefig(os.path.join(save_path,f'H_max{ks}.png'))  # Guarda la figura como archivo PNG
plt.close()  # Cierra la figura para no mostrarla

end = time.perf_counter()

print(f"Elapsed time: {end - start:.4f} seconds")

# Example usage
stop_criteria = stop_criteria
final_area = A
final_lambda = lambda_n
simulation_time = end - start

write_simulation_log(save_path, stop_criteria, final_area, final_lambda, simulation_time)