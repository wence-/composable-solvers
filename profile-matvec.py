from argparse import ArgumentParser
import importlib
import os
import sys
from collections import defaultdict
from functools import partial
import pandas
import numpy as np

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

parser.add_argument("--results-directory",
                    help="Where to put the results")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite existing output?")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--num-matvecs", action="store", default=40,
                    type=int, help="Number of MatVecs to perform")

parser.add_argument("--plot-data", action="store_true",
                    default=False,
                    help="Plot data?  If not passed, collect data")

parser.add_argument("--num-processes", action="store",
                    default=1, type=int,
                    help="Number of processes data was collected on.")

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

result_fmt = "%s_nprocs-%s_size-%s_dimension-%s_degree-%s.h5"


def collect():
    results = os.path.join(os.path.abspath(args.results_directory),
                           result_fmt % (problem.name, problem.u.comm.size,
                                         problem.N, problem.dimension,
                                         problem.degree))

    if os.path.exists(results) and not args.overwrite:
        PETSc.Sys.Print("Not overwriting existing output '%s'\n" % results)
        PETSc.Sys.Print("If you meant to, try with --overwrite\n")
        sys.exit(1)

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
                          map(lambda x: x.petscmat.getInfo(),
                              mat.mat))

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
        store = pandas.HDFStore(results, mode="w")
        for typ in typs:
            store[typ] = pandas.DataFrame(all_data[typ])
            store["info"] = pandas.DataFrame(info)
        store.close()


def plot():
    import glob
    import matplotlib
    matplotlib.use("PDF")
    from matplotlib import pyplot

    output = os.path.join(os.path.abspath(args.results_directory),
                          "%s_dimension-%d-matvec-time.pdf" % (problem.name,
                                                               problem.dimension))
    if os.path.exists(output) and not args.overwrite:
        PETSc.Sys.Print("Not overwriting existing output '%s'\n" % output)
        PETSc.Sys.Print("If you meant to, try with --overwrite\n")
        sys.exit(1)
    result_glob = result_fmt % (problem.name,
                                args.num_processes,
                                problem.N,
                                problem.dimension,
                                "*")
    results = glob.glob(os.path.join(os.path.abspath(args.results_directory),
                                     result_glob))

    datastores = []
    for result in results:
        datastores.append(pandas.HDFStore(result, mode="r"))

    num_mv = datastores[0].aij.matmult["count"]

    degree = np.asarray(map(lambda s: s.info.aij.degree, datastores), dtype=int)
    rows = np.asarray(map(lambda s: s.info.aij.rows, datastores))
    aij_mem = np.asarray(map(lambda s: s.info.aij.memory, datastores))
    matfree_mem = np.asarray(map(lambda s: s.info.matfree.memory, datastores))
    aij_assemble_flops = np.asarray(map(lambda s: s.aij.assemble.flops, datastores))
    aij_matmult_flops = np.asarray(map(lambda s: s.aij.matmult.flops, datastores))
    matfree_assemble_flops = np.asarray(map(lambda s: s.matfree.assemble.flops, datastores))
    matfree_matmult_flops = np.asarray(map(lambda s: s.matfree.matmult.flops, datastores))
    aij_assemble_time = np.asarray(map(lambda s: s.aij.assemble.time, datastores))
    aij_matmult_time = np.asarray(map(lambda s: s.aij.matmult.time, datastores))
    matfree_matmult_time = np.asarray(map(lambda s: s.matfree.matmult.time, datastores))

    fig = pyplot.figure(figsize=(9, 5))
    left = fig.add_subplot(111)

    right = left.twinx()

    left.set_xlabel("Polynomial Degree")

    left.set_ylabel("Time/dof (ms)")
    colours = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']

    left.plot(degree, aij_assemble_time*1e3 / rows,
              linewidth=2, label="Assemble AIJ matrix",
              marker="o",
              color=colours[0])
    left.plot(degree, aij_matmult_time*1e3 / (rows * num_mv),
              linewidth=2, label="Apply AIJ matvec",
              marker="D", color=colours[1])
    left.plot(degree, matfree_matmult_time*1e3 / (rows * num_mv),
              linewidth=2, label="Apply matrix-free matvec",
              marker="^", color=colours[2])

    left.semilogy()
    left.set_xticks(degree)

    left.xaxis.set_ticks_position("bottom")
    right.xaxis.set_ticks_position("bottom")

    right.plot(degree, aij_mem / rows,
               linestyle="dashed",
               marker="o",
               linewidth=2, label="AIJ memory", color=colours[0])

    right.plot(degree, matfree_mem / rows,
               linestyle="dashed",
               marker="D",
               linewidth=2, label="Matrix-free memory", color=colours[1])

    right.set_ylabel("Bytes/dof")

    right.spines["top"].set_visible(False)
    left.spines["top"].set_visible(False)

    handles, labels = left.get_legend_handles_labels()

    h, l = right.get_legend_handles_labels()

    handles = handles + h
    labels = labels + l

    legend = fig.legend(handles, labels,
                        loc=9,
                        bbox_to_anchor=(0.5, 1.1),
                        bbox_transform=fig.transFigure,
                        ncol=3,
                        handlelength=4,
                        fontsize=10,
                        numpoints=1,
                        frameon=False)

    bbox_artists = [legend]
    fig.savefig(output,
                format="pdf",
                bbox_inches="tight",
                bbox_extra_artists=bbox_artists,
                orientation="landscape",
                transparent=True)

    pyplot.close(fig)

    for s in datastores:
        s.close()


if args.plot_data:
    plot()
else:
    collect()
