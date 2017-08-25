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
        size = 200
        degrees = range(1, 5)
        refinements = (2, 1, 0, 0)
    elif prob_args.dimension == 3:
        size = 32
        degrees = range(1, 4)
        refinements = (1, 0, 0)
    else:
        raise ValueError("Unhandled dimension")
elif args.problem == "poisson":
    if prob_args.dimension == 2:
        size = 200
        degrees = (1, 2, 3, 4, 5, 6, 7)
        refinements = (4, 3, 2, 2, 1, 1, 0)
    elif prob_args.dimension == 3:
        size = 22
        degrees = (1, 2, 3, 4, 5)
        refinements = (3, 2, 1, 1, 0)
    else:
        raise ValueError("Unhandled dimension")
else:
    raise ValueError("Unhandled problem %s" % args.problem)

problem = module.Problem(quadrilateral=args.tensor)
problem.N = size
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


first = True

for degree, refinement in zip(degrees, refinements):
    PETSc.Sys.Print("Running degree %d, ref %d" % (degree, refinement))
    problem.reinit(degree=degree, refinements=refinement)

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
                    "assemble_time": assemble_time,
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
