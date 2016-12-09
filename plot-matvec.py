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
                    help="Which problem to plot")

parser.add_argument("--results-file", action="store",
                    default="MatVec-timings.csv",
                    help="Where the results are")

parser.add_argument("--output-file", action="store",
                    default="MatVec-timings.pdf",
                    help="Where to write the output")

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


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem

results = args.results_file

if not os.path.exists(results):
    print "Requested results file '%s' does not exist" % results
    sys.exit(1)


dataframe = pandas.read_csv(results)

dataframe["assemble_per_dof"] = dataframe.assemble_time/dataframe.rows
dataframe["matvec_per_dof"] = dataframe.matmult_time/dataframe.rows
dataframe["bytes_per_dof"] = dataframe.bytes/dataframe.rows

seaborn.set(style="ticks")

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree")

ax.set_ylabel("Time/dof [s]")

ax.set_ylim([dataframe.matvec_per_dof.min()/1.5,
             dataframe.assemble_per_dof.max()*1.5])
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
        sliced = sliced.groupby(["degree"], as_index=False).mean()
        name = {"aij": "AIJ",
                "matfree": "matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        if typ != "matfree":
            ax.plot(sliced.degree, sliced.assemble_per_dof,
                    label="Assemble %s [%s]" % (name, dimstr),
                    linewidth=2, linestyle=linestyle,
                    marker=next(markers),
                    color=next(colors),
                    clip_on=False)

        ax.plot(sliced.degree, sliced.matvec_per_dof,
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

output = args.output_file

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

ax.set_ylim([dataframe.bytes_per_dof.min()/1.5,
             dataframe.bytes_per_dof.max()*1.5])
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
        sliced = sliced.groupby(["degree"], as_index=False).mean()
        name = {"aij": "AIJ",
                "matfree": "Matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        ax.plot(sliced.degree, sliced.bytes_per_dof,
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

output = os.path.join(os.path.abspath(os.path.dirname(output)),
                      "MatVec-memory_%s.pdf" % problem.name)

if os.path.exists(output) and not args.overwrite:
    print "Output PDF '%s' already exists, pass --overwrite to overwrite" % output
    sys.exit(1)

fig.savefig(output, orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])

