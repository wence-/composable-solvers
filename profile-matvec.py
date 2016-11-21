from argparse import ArgumentParser
import importlib
import os
import sys
from collections import defaultdict
from functools import partial
import h5py

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

parser.add_argument("--results-directory",
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


if args.results_directory is None:
    PETSc.Sys.Print("Must provide results directory\n")
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

problem.autorefine = args.autorefine


results = os.path.join(os.path.abspath(args.results_directory),
                       "MatVec-timings_%s.h5" % problem.name)

J = problem.J

assemble_event = PETSc.Log.Event("AssembleMat")
matmult_event = PETSc.Log.Event("MatMult")

timings = defaultdict(partial(defaultdict, dict))
typs = ["aij", "matfree"]
info = {}
if len(problem.function_space) > 1:
    typs.append("nest")

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

    info["rows"] = mat.petscmat.getSize()[0]
    info["cols"] = mat.petscmat.getSize()[1]
    info["degree"] = problem.degree
    return info

for typ in typs:
    # Warmup and allocate
    A = assemble(J, bcs=problem.bcs, mat_type=typ)
    A.force_evaluation()
    Ap = A.petscmat
    x, y = Ap.createVecs()
    Ap.mult(x, y)
    stage = PETSc.Log.Stage("%s matrix" % typ)
    info[typ] = mat_info(A, typ)
    with stage:
        with assemble_event:
            assemble(J, bcs=problem.bcs, mat_type=typ, tensor=A)
            A.force_evaluation()
            Ap = A.petscmat
        for _ in range(args.num_matvecs):
            Ap.mult(x, y)
        timings[typ]["matmult"] = matmult_event.getPerfInfo(stage)
        timings[typ]["assemble"] = assemble_event.getPerfInfo(stage)

def merge_dicts(a, b, datatype):
    result = defaultdict(partial(defaultdict, dict))
    for ka in a:
        for k_ in a[ka]:
            result[ka][k_].update(a[ka][k_])
            for k, v in b[ka][k_].items():
                result[ka][k_][k] += v
    return result

merger = MPI.Op.Create(merge_dicts, commute=True)
all_data = COMM_WORLD.allreduce(timings, op=merger)

for v_ in all_data.values():
    for v in v_.values():
        v["count"] /= COMM_WORLD.size
        v["time"] /= COMM_WORLD.size

if COMM_WORLD.rank == 0:
    if not os.path.exists(os.path.dirname(results)):
        os.makedirs(os.path.dirname(results))
    if args.overwrite:
        mode = "w"
    else:
        mode = "a"
    store = h5py.File(results, mode=mode)

    # Data layout:
    # Multi-D space, coordinate axes are:
    # Nprocs, dimension, degree, problem-size, autorefine, matrix-type
    # At each point, we store (as attributes)
    #   - bytes, row, cols for the matrix
    #   - time, flops, call-count for assembly and matmult
    base_group = "/%d/%d/%d/%d/%s" % (COMM_WORLD.size, problem.dimension,
                                      problem.degree, problem.N,
                                      problem.autorefine)

    if base_group not in store:
        base_group = store.create_group(base_group)
    else:
        base_group = store[base_group]
    for typ in typs:
        if typ not in base_group:
            group = base_group.create_group(typ)
        else:
            group = base_group[typ]
        group.attrs["bytes"] = info[typ]["memory"]
        group.attrs["rows"] = info[typ]["rows"]
        group.attrs["cols"] = info[typ]["cols"]
        for event in ["assemble", "matmult"]:
            if event not in group:
                g = group.create_group(event)
            else:
                g = group[event]
            data = all_data[typ][event]
            g.attrs["time"] = data["time"]
            g.attrs["flops"] = data["flops"]
            g.attrs["count"] = data["count"]
    store.close()
