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

parser.add_argument("--autorefine", action="store_true", default=False,
                    help="Refine meshes to give approximately fixed number of dofs?")

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

problem.autorefine = args.autorefine

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
                          "%s_nprocs-%d_dimension-%d-matvec-time.pdf"
                          % (problem.name, args.num_processes,
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

    types = datastores[0].info.keys()
    degree = np.asarray(map(lambda s: s.info.aij.degree, datastores), dtype=int)
    rows = np.asarray(map(lambda s: s.info.aij.rows, datastores))

    mem = {}
    assemble_flops = {}
    matmult_flops = {}
    assemble_time = {}
    matmult_time = {}
    for typ in types:
        mem[typ] = np.asarray(map(lambda s: s.info[typ].memory, datastores))
        assemble_flops[typ] = np.asarray(map(lambda s: s[typ].assemble.flops,
                                             datastores))
        matmult_flops[typ] = np.asarray(map(lambda s: s[typ].matmult.flops,
                                            datastores))
        assemble_time[typ] = np.asarray(map(lambda s: s[typ].assemble.time,
                                            datastores))
        matmult_time[typ] = np.asarray(map(lambda s: s[typ].matmult.time,
                                           datastores))
    fig = pyplot.figure(figsize=(9, 5))
    left = fig.add_subplot(111)

    right = left.twinx()

    left.set_xlabel("Polynomial Degree")

    left.set_ylabel("Time/dof (ms)")
    colours = ['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b']
    markers = ["o", "D", "^", "v", "p"]

    cit = iter(colours)
    mit = iter(markers)
    left.plot(degree, assemble_time["aij"]*1e3 / rows,
              linewidth=2, label="Assemble AIJ matrix",
              marker=next(mit),
              color=next(cit))
    if "nest" in types:
        left.plot(degree, assemble_time["nest"]*1e3 / rows,
                  linewidth=2, label="Assemble nest matrix",
                  marker=next(mit),
                  color=next(cit))
    left.plot(degree, matmult_time["aij"]*1e3 / (rows * num_mv),
              linewidth=2, label="AIJ MatVec",
              marker=next(mit),
              color=next(cit))
    if "nest" in types:
        left.plot(degree, matmult_time["nest"]*1e3 / (rows * num_mv),
                  linewidth=2, label="Nest MatVec",
                  marker=next(mit),
                  color=next(cit))

    left.plot(degree, matmult_time["matfree"]*1e3 / (rows * num_mv),
              linewidth=2, label="Matrix-free MatVec",
              marker=next(mit),
              color=next(cit))

    left.semilogy()
    left.set_xticks(degree)

    left.xaxis.set_ticks_position("bottom")
    right.xaxis.set_ticks_position("bottom")

    it = iter(colours)
    mit = iter(markers)
    right.plot(degree, mem["aij"] / rows,
               linestyle="dashed",
               marker=next(mit),
               linewidth=2, label="AIJ memory", color=next(it))

    if "nest" in types:
        right.plot(degree, mem["nest"] / rows,
                   linestyle="dashed",
                   marker=next(mit),
                   linewidth=2, label="Nest memory", color=next(it))

    right.plot(degree, mem["matfree"] / rows,
               linestyle="dashed",
               marker=next(mit),
               linewidth=2, label="Matrix-free memory", color=next(it))

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
                        ncol=2,
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
