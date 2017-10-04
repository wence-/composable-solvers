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
    print("Requested data file '%s' does not exist" % data)
    sys.exit(1)


dataframe = pandas.read_csv(data)

dataframe["assemble_dof_per_second"] = (dataframe.rows/dataframe.assemble_time)
dataframe["matvec_dof_per_second"] = (dataframe.rows/dataframe.matmult_time)
dataframe["bytes_per_dof"] = dataframe.bytes/dataframe.rows

seaborn.set(style="ticks")

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)

ax.set_xlabel("Polynomial degree\n(2D dofs/process)\n(3D dofs/process)", fontsize=FONTSIZE)

ax.set_ylabel("dofs/second", fontsize=FONTSIZE)

ax.set_ylim([dataframe.assemble_dof_per_second.min()/1.5,
             dataframe.matvec_dof_per_second.max()*1.5])
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
            ax.plot(sliced.degree, sliced.assemble_dof_per_second,
                    label="Assemble %s [%s]" % (name, dimstr),
                    linewidth=LINEWIDTH, linestyle=linestyle,
                    markersize=MARKERSIZE,
                    marker=next(markers),
                    color=next(colors),
                    clip_on=False)

        ax.plot(sliced.degree, sliced.matvec_dof_per_second,
                label="MatMult %s [%s]" % (name, dimstr),
                linewidth=LINEWIDTH, linestyle=linestyle,
                markersize=MARKERSIZE,
                marker=next(markers),
                color=next(colors),
                clip_on=False)


def doflabel(dofs):
    if dofs > 1e6:
        return '%.1fM' % (dofs/(1e6))
    elif dofs > 1e3:
        return '%dk' % (dofs/(1e3))
    return '%d' % (dofs)


sliced = dataframe.loc[lambda df: df.type == "aij"].loc[lambda df: df.num_processes == num_processes]

sliced = sliced.groupby(["degree", "dimension"], as_index=False).mean()

twod_dofs = ["(%s)" % doflabel(d/num_processes) for d in sliced.loc[lambda df: df.dimension == 2].rows]
threed_dofs = ["(%s)" % doflabel(d/num_processes) for d in sliced.loc[lambda df: df.dimension == 3].rows]
for _ in range(len(twod_dofs) - len(threed_dofs)):
    threed_dofs.append("")

ax.set_xticks(list(range(1, 1+len(twod_dofs))))
ax.set_xticklabels(["%s\n%s\n%s" % (d+1, twod_dofs[d], threed_dofs[d]) for d in range(len(twod_dofs))])

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

