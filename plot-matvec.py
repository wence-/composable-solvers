from argparse import ArgumentParser
import importlib
import os
import sys
import numpy
import pandas
import matplotlib as mpl
import math
from matplotlib import pyplot
import seaborn

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
                       "MatVec-timings_%s.csv" % problem.name)

if not os.path.exists(results):
    print "Requested results file '%s' does not exist" % results
    sys.exit(1)


dataframe = pandas.read_csv(results)

seaborn.set(style="ticks")

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree")

ax.set_ylabel("Time/dof [s]")

ax.semilogy()

linestyles = iter(["solid", "dashed"])
colours = ("#000000", "#E69F00", "#56B4E9", "#009E73",
           "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
for dim in dataframe["dimension"].drop_duplicates():
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(colours)
    for typ in sorted(dataframe["type"].drop_duplicates()):
        sliced = (dataframe.loc[lambda df: df.type == typ]
                  .loc[lambda df: df.dimension == dim]
                  .loc[lambda df: df.num_processes == args.num_processes])
        name = {"aij": "AIJ",
                "matfree": "matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        if typ != "matfree":
            ax.plot(sliced.degree, (sliced.assemble_time / sliced.rows),
                    label="Assemble %s [%s]" % (name, dimstr),
                    linewidth=2, linestyle=linestyle,
                    marker=next(markers),
                    color=next(colors),
                    clip_on=False)

        ax.plot(sliced.degree, (sliced.matmult_time / sliced.rows),
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

seaborn.despine(fig)

output = os.path.join(os.path.abspath(args.results_directory),
                      "MatVec-timings_%s.pdf" % problem.name)

if os.path.exists(output) and not args.overwrite:
    print "Output PDF '%s' already exists, pass --overwrite to overwrite" % output
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
for dim in dataframe["dimension"].drop_duplicates():
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(colours)
    for typ in sorted(dataframe["type"].drop_duplicates()):
        sliced = (dataframe.loc[lambda df: df.type == typ]
                  .loc[lambda df: df.dimension == dim]
                  .loc[lambda df: df.num_processes == args.num_processes])
        name = {"aij": "AIJ",
                "matfree": "Matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        ax.plot(sliced.degree, (sliced.bytes / sliced.rows),
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

seaborn.despine(fig)

output = os.path.join(os.path.abspath(args.results_directory),
                      "MatVec-memory_%s.pdf" % problem.name)

if os.path.exists(output) and not args.overwrite:
    print "Output PDF '%s' already exists, pass --overwrite to overwrite" % output
    sys.exit(1)

fig.savefig(output, orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])

