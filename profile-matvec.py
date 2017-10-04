from argparse import ArgumentParser
import importlib
import os
import sys
import pandas
from firedrake import assemble, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI
from functools import reduce

PETSc.Log.begin()

parser = ArgumentParser(description="""Profile matvecs""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--tensor", action="store_true", default=False,
                    help="Use tensor-product cells?")

parser.add_argument("--output-file", action="store",
                    default="MatVec-timings.csv",
                    help="Where to put the results")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?  Default is to append.")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--num-matvecs", action="store", default=40,
                    type=int, help="Number of MatVecs to perform")

args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


if args.problem is None:
    PETSc.Sys.Print("Must provide problem type\n")
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
prob_args, _ = module.Problem.argparser().parse_known_args()

if args.problem == "rayleigh_benard":
    if prob_args.dimension == 2:
        sizes = (150, 200, 250, 200)
        degrees = range(1, 5)
        refinements = (3, 2, 1, 1)
    elif prob_args.dimension == 3:
        sizes = (40, 50, 29)
        degrees = range(1, 4)
        refinements = (1, 0, 0)
    else:
        raise ValueError("Unhandled dimension")
elif args.problem == "poisson":
    if prob_args.dimension == 2:
        sizes =       (200, 200, 150, 200, 160, 130, 250)
        degrees =     (1, 2, 3, 4, 5, 6, 7)
        refinements = (4, 3, 3, 2, 2, 2, 1)
    elif prob_args.dimension == 3:
        sizes =       (50, 50, 40, 30, 50)
        degrees =     (1, 2, 3, 4, 5)
        refinements = (2, 1, 1, 1, 0)
    else:
        raise ValueError("Unhandled dimension")
else:
    raise ValueError("Unhandled problem %s" % args.problem)

problem = module.Problem(quadrilateral=args.tensor)
results = os.path.abspath(args.output_file)


def mat_info(mat, typ):
    if typ == "matfree":
        ctx = mat.petscmat.getPythonContext()
        info = ctx.getInfo(mat.petscmat)
        info["nz_used"] = 0
    elif typ == "aij":
        info = mat.petscmat.getInfo()
    elif typ == "nest":
        info = reduce(lambda x, y: dict((k, x[k] + y[k]) for k in x),
                      map(lambda x: x.handle.getInfo(),
                          mat.M))

    rows = mat.petscmat.getSize()[0]
    cols = mat.petscmat.getSize()[1]
    bytes = info["memory"]
    nz = info["nz_used"]
    return rows, cols, bytes, nz


first = True
workaround_flop_counting_bug = True

if workaround_flop_counting_bug:
    # Prior to c91eb2e, PyOP2 overcounted flops by this factor
    scaling = 3.0
else:
    scaling = 1.0


sizeof_int = PETSc.IntType().dtype.itemsize
sizeof_double = PETSc.ScalarType().dtype.itemsize


PETSc.Sys.Print("Int Type has %d bytes, Scalar Type has %d bytes" %
                (sizeof_int, sizeof_double))

def aij_matvec_bytes(rows, cols, nz, rbs=1, cbs=1):
    # Gropp et al. 2000
    if rbs == cbs and rbs != 1:
        pass
    else:
        rbs = 1
        cbs = 1
    return ((cols + rows)*sizeof_double  # Vec read/write
            + (rows / rbs)*sizeof_int    # Row pointer
            + (nz / rbs**2)*sizeof_int   # col idx
            + nz*sizeof_double)          # nonzeros


def aij_matvec_flops(nz):
    return float(2*nz)


def aij_matvec_ai(rows, cols, nz):
    return aij_matvec_flops(nz) / aij_matvec_bytes(rows, cols, nz)


def nest_matvec_bytes(mat):
    nbyte = 0
    for m in mat.M:
        m = m.handle
        rbs, cbs = m.getBlockSizes()
        rows, cols = m.getSize()
        nz = m.getInfo()["nz_used"]
        nbyte += aij_matvec_bytes(rows, cols, nz, rbs, cbs)
    return nbyte


def nest_matvec_ai(mat, nz):
    return aij_matvec_flops(nz) / nest_matvec_bytes(mat)


def aij_assemble_ai(rows, row_dof_per_cell,
                    cols, col_dof_per_cell,
                    coords, coord_dof_per_cell,
                    ncell, nz, flops):
    field_bytes = coords*sizeof_double
    # RW for mat data since we increment
    mat_bytes = nz*sizeof_int + 2*nz*sizeof_double + rows*sizeof_int
    map_bytes = (row_dof_per_cell + col_dof_per_cell + coord_dof_per_cell)*sizeof_int*ncell
    return float(flops) / (field_bytes + mat_bytes + map_bytes)


def matfree_matvec_bytes(rows, row_dof_per_cell,
                         cols, col_dof_per_cell,
                         coords, coord_dof_per_cell,
                         ncell):
    # Perfect cache.
    # field data (RW for output since we increment)
    field_bytes = (rows*2 + cols + coords)*sizeof_double
    # indirect data
    map_bytes = (row_dof_per_cell
                 + col_dof_per_cell
                 + coord_dof_per_cell)*ncell*sizeof_int
    return field_bytes + map_bytes


def matfree_matvec_flops(flops):
    return float(flops)


def matfree_matvec_ai(rows, row_dof_per_cell,
                      cols, col_dof_per_cell,
                      coords, coord_dof_per_cell,
                      ncell, flops):
    return matfree_matvec_flops(flops) / matfree_matvec_bytes(rows, row_dof_per_cell,
                                                              cols, col_dof_per_cell,
                                                              coords, coord_dof_per_cell,
                                                              ncell)

for size, degree, refinement in zip(sizes, degrees, refinements):
    PETSc.Sys.Print("Running degree %d, size %d, ref %d" % (degree, size, refinement))
    problem.reinit(size=size, degree=degree, refinements=refinement)

    J = problem.J

    assemble_event = PETSc.Log.Event("AssembleMat")
    matmult_event = PETSc.Log.Event("MatMult")

    typs = ["aij", "matfree"]
    if len(problem.function_space) > 1:
        typs.append("nest")

    num_cells = problem.comm.allreduce(problem.mesh.cell_set.size, op=MPI.SUM)

    for typ in typs:
        # Warmup and allocate
        A = assemble(J, bcs=problem.bcs, mat_type=typ)
        A.force_evaluation()
        Ap = A.petscmat
        x, y = Ap.createVecs()
        x.setRandom()
        Ap.mult(x, y)

        stage = PETSc.Log.Stage("P(%d) %s matrix" % (degree, typ))
        with stage:
            with assemble_event:
                assemble(J, bcs=problem.bcs, mat_type=typ, tensor=A)
                A.force_evaluation()
                Ap = A.petscmat
            for _ in range(args.num_matvecs):
                x.set(_)
                y.set(_)
                Ap.mult(x, y)

            matmult = matmult_event.getPerfInfo()
            assembly = assemble_event.getPerfInfo()
            matmult_time = problem.comm.allreduce(matmult["time"], op=MPI.MAX) / (args.num_matvecs)
            if typ == "matfree":
                matmult_flops = problem.comm.allreduce(matmult["flops"], op=MPI.SUM) / (args.num_matvecs * scaling)
            else:
                matmult_flops = problem.comm.allreduce(matmult["flops"], op=MPI.SUM) / args.num_matvecs
            assemble_time = problem.comm.allreduce(assembly["time"], op=MPI.MAX)
            assemble_flops = problem.comm.allreduce(assembly["flops"], op=MPI.SUM) / scaling

        rows, cols, bytes, nz = mat_info(A, typ)

        V = problem.function_space
        Vc = problem.mesh.coordinates.function_space()
        if typ == "matfree":
            ai = matfree_matvec_ai(rows, V.cell_node_map().arity,
                                   cols, V.cell_node_map().arity,
                                   Vc.dof_dset.layout_vec.getSizes()[-1],
                                   Vc.cell_node_map().arity,
                                   num_cells,
                                   matmult_flops)
            assemble_ai = 0
        elif typ == "aij":
            ai = aij_matvec_ai(rows, cols, nz)
            assemble_ai = aij_assemble_ai(rows, V.cell_node_map().arity,
                                          cols, V.cell_node_map().arity,
                                          Vc.dof_dset.layout_vec.getSizes()[-1],
                                          Vc.cell_node_map().arity,
                                          num_cells,
                                          nz,
                                          assemble_flops)
        elif typ == "nest":
            ai = nest_matvec_ai(A, nz)
            assemble_ai = aij_assemble_ai(rows, V.cell_node_map().arity,
                                          cols, V.cell_node_map().arity,
                                          Vc.dof_dset.layout_vec.getSizes()[-1],
                                          Vc.cell_node_map().arity,
                                          num_cells,
                                          nz,
                                          assemble_flops)

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            if args.overwrite:
                if first:
                    mode = "w"
                    header = True
                else:
                    mode = "a"
                    header = False
                first = False
            else:
                mode = "a"
                header = not os.path.exists(results)

            data = {"rows": rows,
                    "cols": cols,
                    "type": typ,
                    "bytes": bytes,
                    "nonzeros": nz,
                    "assemble_time": assemble_time,
                    "matvec_ai": ai,
                    "assemble_ai": assemble_ai,
                    "assemble_flops": assemble_flops,
                    "matmult_time": matmult_time,
                    "matmult_flops": matmult_flops,
                    "mesh_size": num_cells,
                    "dimension": problem.dimension,
                    "degree": problem.degree,
                    "num_processes": problem.comm.size,
                    "problem": problem.name,
                    "cell_type": {True: "tensor", False: "simplex"}[args.tensor]}
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(results, index=False, mode=mode, header=header)
