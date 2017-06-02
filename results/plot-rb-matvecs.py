import os
import sys
import pandas
from matplotlib import pyplot
import seaborn


FONTSIZE = 16
MARKERSIZE = 12
LINEWIDTH = 3
data = "rb-matvecs.csv"
num_processes = 48

if not os.path.exists(data):
    print "Requested data file '%s' does not exist" % data
    sys.exit(1)


dataframe = pandas.read_csv(data)

dataframe["assemble_per_dof"] = dataframe.assemble_time/dataframe.rows
dataframe["matvec_per_dof"] = dataframe.matmult_time/dataframe.rows
dataframe["bytes_per_dof"] = dataframe.bytes/dataframe.rows

seaborn.set(style="ticks")

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree of scalar space", fontsize=FONTSIZE)

ax.set_ylabel("Time/dof [s]", fontsize=FONTSIZE)

ax.set_ylim([dataframe.matvec_per_dof.min()/1.5,
             dataframe.assemble_per_dof.max()*1.5])
ax.semilogy()

ax.set_xticks([1, 2, 3, 4, 5])
linestyles = iter(["solid", "dashed"])
colours = seaborn.color_palette(n_colors=6)

for dim in dataframe["dimension"].drop_duplicates():
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(colours)
    for typ in sorted(dataframe["type"].drop_duplicates()):
        sliced = (dataframe.loc[lambda df: df.type == typ]
                  .loc[lambda df: df.dimension == dim]
                  .loc[lambda df: df.num_processes == num_processes])
        sliced = sliced.groupby(["degree"], as_index=False).mean()
        name = {"aij": "AIJ",
                "matfree": "matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        if typ != "matfree":
            ax.plot(sliced.degree, sliced.assemble_per_dof,
                    label="Assemble %s [%s]" % (name, dimstr),
                    linewidth=LINEWIDTH, linestyle=linestyle,
                    markersize=MARKERSIZE,
                    marker=next(markers),
                    color=next(colors),
                    clip_on=False)

        ax.plot(sliced.degree, sliced.matvec_per_dof,
                label="MatMult %s [%s]" % (name, dimstr),
                linewidth=LINEWIDTH, linestyle=linestyle,
                markersize=MARKERSIZE,
                marker=next(markers),
                color=next(colors),
                clip_on=False)

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)
for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)
handles, labels = ax.get_legend_handles_labels()

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.25),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=4,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)

fig.savefig("rb-matvec-time.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree of scalar space", fontsize=FONTSIZE)

ax.set_ylabel("Bytes/dof", fontsize=FONTSIZE)

ax.set_ylim([dataframe.bytes_per_dof.min()/1.5,
             dataframe.bytes_per_dof.max()*1.5])
ax.semilogy()
ax.set_xticks([1, 2, 3, 4, 5])

linestyles = iter(["solid", "dashed"])
for dim in dataframe["dimension"].drop_duplicates():
    linestyle = next(linestyles)
    markers = iter(["o", "s", "^", "D", "v"])
    colors = iter(colours)
    for typ in sorted(dataframe["type"].drop_duplicates()):
        sliced = (dataframe.loc[lambda df: df.type == typ]
                  .loc[lambda df: df.dimension == dim]
                  .loc[lambda df: df.num_processes == num_processes])
        sliced = sliced.groupby(["degree"], as_index=False).mean()
        name = {"aij": "AIJ",
                "matfree": "Matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]
        ax.plot(sliced.degree, sliced.bytes_per_dof,
                label="%s [%s]" % (name, dimstr),
                linewidth=LINEWIDTH, linestyle=linestyle,
                markersize=MARKERSIZE,
                marker=next(markers),
                color=next(colors),
                clip_on=False)

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)
for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)
handles, labels = ax.get_legend_handles_labels()

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.25),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=4,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)

fig.savefig("rb-matvec-memory.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])

