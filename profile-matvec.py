from argparse import ArgumentParser
import importlib
import os
import sys
import pandas
from firedrake import assemble, COMM_WORLD
from firedrake.petsc import PETSc
from mpi4py import MPI

PETSc.Log.begin()

parser = ArgumentParser(description="""Profile matvecs""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--autorefine", action="store_true", default=False,
                    help="Refine meshes to give approximately fixed number of dofs?")

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

if prob_args.dimension == 2:
    degrees = range(1, 8)
elif prob_args.dimension == 3:
    degrees = range(1, 6)
else:
    raise ValueError("Unhandled dimension")

problem = module.Problem()

problem.autorefine = args.autorefine
results = os.path.abspath(args.output_file)


def mat_info(mat, typ):
    if typ == "matfree":
        ctx = mat.petscmat.getPythonContext()
        info = ctx.getInfo(mat.petscmat)
    elif typ == "aij":
        info = mat.petscmat.getInfo()
    elif typ == "nest":
        info = reduce(lambda x, y: dict((k, x[k] + y[k]) for k in x),
                      map(lambda x: x.handle.getInfo(),
                          mat.M))

    rows = mat.petscmat.getSize()[0]
    cols = mat.petscmat.getSize()[1]
    bytes = info["memory"]
    return rows, cols, bytes


for degree in degrees:
    PETSc.Sys.Print("Running degree %d" % degree)
    problem.reinit(degree=degree)

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
        Ap.mult(x, y)
        stage = PETSc.Log.Stage("P(%d) %s matrix" % (degree, typ))
        with stage:
            with assemble_event:
                assemble(J, bcs=problem.bcs, mat_type=typ, tensor=A)
                A.force_evaluation()
                Ap = A.petscmat
            for _ in range(args.num_matvecs):
                Ap.mult(x, y)

            matmult = matmult_event.getPerfInfo()
            assembly = assemble_event.getPerfInfo()
            matmult_time = problem.comm.allreduce(matmult["time"], op=MPI.SUM) / (problem.comm.size * args.num_matvecs)
            matmult_flops = problem.comm.allreduce(matmult["flops"], op=MPI.SUM) / args.num_matvecs
            assemble_time = problem.comm.allreduce(assembly["time"], op=MPI.SUM) / problem.comm.size
            assemble_flops = problem.comm.allreduce(assembly["flops"], op=MPI.SUM)

        rows, cols, bytes = mat_info(A, typ)

        if COMM_WORLD.rank == 0:
            if not os.path.exists(os.path.dirname(results)):
                os.makedirs(os.path.dirname(results))

            if args.overwrite:
                mode = "w"
                header = True
            else:
                mode = "a"
                header = not os.path.exists(results)

            data = {"rows": rows,
                    "cols": cols,
                    "type": typ,
                    "bytes": bytes,
                    "assemble_time": assemble_time,
                    "assemble_flops": assemble_flops,
                    "matmult_time": matmult_time,
                    "matmult_flops": matmult_flops,
                    "mesh_size": num_cells,
                    "dimension": problem.dimension,
                    "degree": problem.degree,
                    "num_processes": problem.comm.size,
                    "problem": problem.name}
            df = pandas.DataFrame(data, index=[0])
            df.to_csv(results, index=False, mode=mode, header=header)
