from argparse import ArgumentParser
import importlib
import os
import sys
import h5py
import numpy
import xarray
import matplotlib as mpl
import math
from matplotlib import pyplot

parser = ArgumentParser(description="""Plot matvec profiles""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to profile")

parser.add_argument("--results-directory",
                    help="Where the results are")

parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Overwrite output plot?")

parser.add_argument("--help", action="store_true",
                    help="Show help")

parser.add_argument("--num-processes", action="store",
                    default=1, type=int,
                    help="Number of processes data was collected on.")

parser.add_argument("--autorefine", action="store_true",
                    default=False,
                    help="Were meshes refined to give approx same number of dofs?")

parser.add_argument("--twod-size", action="store",
                    type=int,
                    help="Size of 2D problem")

parser.add_argument("--threed-size", action="store",
                    type=int,
                    help="Size of 3D problem")

args, _ = parser.parse_known_args()


if args.help:
    parser.print_help()


if args.problem is None:
    print "Must provide problem type"
    sys.exit(1)


if args.results_directory is None:
    print "Must provide results directory"
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

results = os.path.join(os.path.abspath(args.results_directory),
                       "MatVec-timings_%s.h5" % problem.name)

if not os.path.exists(results):
    print "Requested results file '%s' does not exist" % results
    sys.exit(1)

store = h5py.File(results, mode="r")

# Get some layout data:
num_processes = sorted(map(int, store.keys()))
degrees = set()
types = set()
dimensions = set()
for proc in store.keys():
    base_group = store[proc]
    dimensions.update(map(int, base_group.keys()))
    for dimension in base_group.keys():
        degrees.update(map(int, base_group[dimension].keys()))
        for degree in base_group[dimension].keys():
            group = base_group[dimension][degree]
            N = {2: args.twod_size,
                 3: args.threed_size}[int(dimension)]
            path = "%d/%s" % (N, args.autorefine)
            if path not in group:
                print "Data for '%s' not found" % path
                sys.exit(1)
            types.update(group[path].keys())

shape = (len(num_processes), len(dimensions), len(degrees), len(types))
labels = ['num_processes', 'dimension', 'degree', 'type']

dataset = xarray.Dataset(coords={'num_processes': num_processes,
                                 'dimension': sorted(dimensions),
                                 'degree': sorted(degrees),
                                 'type': sorted(types)},
                         data_vars={'bytes': (labels,
                                              numpy.full(shape, numpy.nan)),
                                    'rows': (labels,
                                             numpy.full(shape, numpy.nan)),
                                    'cols': (labels,
                                             numpy.full(shape, numpy.nan)),
                                    'assemble_time': (labels,
                                                      numpy.full(shape, numpy.nan)),
                                    'assemble_flops': (labels,
                                                       numpy.full(shape, numpy.nan)),
                                    'assemble_count': (labels,
                                                       numpy.full(shape, numpy.nan)),
                                    'matmult_time': (labels,
                                                     numpy.full(shape, numpy.nan)),
                                    'matmult_flops': (labels,
                                                      numpy.full(shape, numpy.nan)),
                                    'matmult_count': (labels,
                                                      numpy.full(shape, numpy.nan))})

# Convert data to xarray
for nproc in store.keys():
    base_group = store[nproc]
    for dimension in base_group.keys():
        group = base_group[dimension]
        for degree in group.keys():
            g = group[degree]
            N = {2: args.twod_size,
                 3: args.threed_size}[int(dimension)]
            path = "%d/%s" % (N, args.autorefine)
            g = g[path]
            types = g.keys()
            for typ in types:
                metadata = g[typ].attrs
                slice = dict(num_processes=int(nproc),
                             dimension=int(dimension),
                             degree=int(degree),
                             type=typ)
                dataset.bytes.loc[slice] = metadata["bytes"]
                dataset.rows.loc[slice] = metadata["rows"]
                dataset.cols.loc[slice] = metadata["cols"]
                for event in ["assemble", "matmult"]:
                    data = g[typ][event].attrs
                    for v in ["count", "time", "flops"]:
                        key = "%s_%s" % (event, v)
                        dataset[key].loc[slice] = data[v]


class BetterLogFormatter(mpl.ticker.Formatter):

    def __call__(self, x, pos=None):
        b = 10

        if x == 0:
            return "$\\mathdefault{0}$"

        expt = math.log(abs(x), b)
        dec = mpl.ticker.is_close_to_int(expt)

        sgn = '-' if x < 0 else ''
        if dec:
            return "$\\mathdefault{%s%d^{%.2g}}$" % (sgn, b, expt)
        else:
            expt = int(expt)
            mult = x / (b**expt)
            return "$\\mathdefault{%s%.1g\\times\,%d^{%.2g}}$" % (sgn, mult, b, expt)


fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree")

ax.set_ylabel("Time/dof (ms)")

ax.semilogy()

linestyles = iter(["solid", "dashed"])
for dim in dataset.coords["dimension"].values:
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'])
    for typ in dataset.coords["type"].values:
        slice = dict(dimension=dim, num_processes=1,
                     type=typ)
        sliced = dataset.loc[slice]
        name = {"aij": "AIJ",
                "matfree": "matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        if typ != "matfree":
            ax.plot(dataset.degree, (sliced.assemble_time * 1e3) / sliced.rows,
                    label="Assemble %s [%s]" % (name, dimstr),
                    linewidth=2, linestyle=linestyle,
                    marker=next(markers),
                    color=next(colors),
                    clip_on=False)

        ax.plot(dataset.degree, (sliced.matmult_time * 1e3) / (sliced.matmult_count * sliced.rows),
                label="MatMult %s [%s]" % (name, dimstr),
                linewidth=2, linestyle=linestyle,
                marker=next(markers),
                color=next(colors),
                clip_on=False)

handles, labels = ax.get_legend_handles_labels()

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=4,
                    fontsize=9,
                    numpoints=1,
                    frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

ax.yaxis.set_major_formatter(BetterLogFormatter())

output = os.path.join(os.path.abspath(args.results_directory),
                      "MatVec-timings_%s.pdf" % problem.name)

if os.path.exists(output) and not args.overwrite:
    print "Output PDF '%s' already exists, pass --overwrite to overwrite" % results
    sys.exit(1)

fig.savefig(output, orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree")

ax.set_ylabel("Bytes/dof")


ax.semilogy()

linestyles = iter(["solid", "dashed"])
for dim in dataset.coords["dimension"].values:
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(['#30a2da', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b'])
    for typ in dataset.coords["type"].values:
        slice = dict(dimension=dim, num_processes=1,
                     type=typ)
        sliced = dataset.loc[slice]
        name = {"aij": "AIJ",
                "matfree": "Matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        ax.plot(dataset.degree, (sliced.bytes / sliced.rows),
                label="%s [%s]" % (name, dimstr),
                linewidth=2, linestyle=linestyle,
                marker=next(markers),
                color=next(colors),
                clip_on=False)

handles, labels = ax.get_legend_handles_labels()

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=4,
                    fontsize=9,
                    numpoints=1,
                    frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

ax.yaxis.set_major_formatter(BetterLogFormatter())

output = os.path.join(os.path.abspath(args.results_directory),
                      "MatVec-memory_%s.pdf" % problem.name)

if os.path.exists(output) and not args.overwrite:
    print "Output PDF '%s' already exists, pass --overwrite to overwrite" % results
    sys.exit(1)

fig.savefig(output, orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])
