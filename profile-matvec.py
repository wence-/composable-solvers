# flake8: noqa: F403
from argparse import ArgumentParser
import importlib, sys
import os
from collections import defaultdict
from functools import partial
import pandas

from firedrake import *
from firedrake.petsc import PETSc
from mpi4py import MPI
parameters["pyop2_options"]["loop_fusion"] = True
PETSc.Log.begin()
from pyop2.profiling import timed_stage, timed_region

parser = ArgumentParser(description="""Profile matvecs""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--output-directory",
                    help="Where to put the results")

parser.add_argument("--overwrite-output", action="store_true", default=False,
                    help="Overwrite existing output?")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--max-matvecs", action="store", default=100,
                    type=int, help="Maximum number of MatVecs to perform, data will be gathered for all range(1,Max, 10)")

args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)


if args.problem is None:
    PETSc.Sys.Print("Must provide problem type\n")
    sys.exit(1)


if args.output_directory is None:
    PETSc.Sys.Print("Must provide output directory\n")
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

results = os.path.join(os.path.abspath(args.output_directory),
                       problem.datastore_name)
if os.path.exists(results) and not args.overwrite_output:
    PETSc.Sys.Print("Not overwriting existing output '%s'\n" % results)
    PETSc.Sys.Print("If you meant to, try with --overwrite-output\n")
    sys.exit(1)
    
J = problem.J

assemble_event = PETSc.Log.Event("AssembleMat")

timings = defaultdict(partial(defaultdict, dict))
typs = ["aij", "matfree"]
if len(problem.function_space) > 1:
    typs.append("nest")

for typ in typs:
    # Warmup and allocate
    A = assemble(J, bcs=problem.bcs, mat_type=typ)
    A.force_evaluation()
    Ap = A.petscmat
    x, y = Ap.createVecs()
    Ap.mult(x, y)
    stage = PETSc.Log.Stage("%s matrix" % typ)
    with stage:
        with assemble_event:
            assemble(J, bcs=problem.bcs, mat_type=typ, tensor=A)
            A.force_evaluation()
            Ap = A.petscmat
        for nmv in range(1, args.max_matvecs + 1, 10):
            event = PETSc.Log.Event("MatMult-%d" % nmv)
            with event:
                for i in range(nmv):
                    Ap.mult(x, y)
            timings[typ]["matmult-%d" % nmv] = event.getPerfInfo(stage)
            timings[typ]["matmult-%d" % nmv]["calls"] = nmv
        timings[typ]["assemble"] = assemble_event.getPerfInfo(stage)
        timings[typ]["assemble"]["calls"] = 1

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
        v["calls"] /= COMM_WORLD.size

# print timings


def to_dataframe(data):
    assemble = data["assemble"]
    matmult = {}
    for k, v in data.items():
        if k == "assemble":
            continue
        matmult[v["calls"]] = {}
        matmult[v["calls"]].update(assemble)
        for k_, v_ in v.items():
            matmult[v["calls"]][k_] += v_
        matmult[v["calls"]].pop("calls")
    return pandas.DataFrame(matmult).T

if COMM_WORLD.rank == 0:
    if not os.path.exists(os.path.dirname(results)):
        os.makedirs(os.path.dirname(results))
    store = pandas.HDFStore(results, mode="w")
    for typ in typs:
        store[typ] = to_dataframe(all_data[typ])
    store.close()
