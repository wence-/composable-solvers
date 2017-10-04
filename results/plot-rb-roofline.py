import os
import sys
import numpy
import pandas
from matplotlib import pyplot
import seaborn


FONTSIZE = 16
MARKERSIZE = 12
data = "rb-matvecs.csv"

num_processes = 48
if not os.path.exists(data):
    print("Requested data file '%s' does not exist" % data)
    sys.exit(1)

dataframe = pandas.read_csv(data)

PEAK_BW = 119.4 * (num_processes // 24)      # GB/s
STREAM_TRIAD = 74.1 * (num_processes // 24)  # GB/s
PEAK_FLOPS = 518.4 * (num_processes // 24)   # GFLOP/s

seaborn.set(style="ticks")

fig = pyplot.figure(figsize=(9, 5), frameon=False)
ax = fig.add_subplot(111)
ax.set_xscale("log", basex=2)
ax.set_yscale("log", basey=2)

ax.set_xlabel("Arithmetic intensity [flop/byte]", fontsize=FONTSIZE)
ax.set_ylabel("Gflop/s", fontsize=FONTSIZE)

ax.set_xlim([2**-4, 2**8])
ax.set_ylim([2**2, 2**11])


def add_roofline(BW, FLOPS):
    xes = [2**n for n in range(-4, 9)]
    xes = numpy.insert(xes, numpy.searchsorted(xes, FLOPS/BW), FLOPS/BW)
    yes = [min(FLOPS, BW*x) for x in xes]
    ax.plot(xes, yes, linewidth=2, color="grey", zorder=1)


add_roofline(PEAK_BW, PEAK_FLOPS)
add_roofline(STREAM_TRIAD, PEAK_FLOPS)
add_roofline(PEAK_BW, PEAK_FLOPS / 2)
add_roofline(PEAK_BW, PEAK_FLOPS / 8)
props = dict(facecolor='white', alpha=0.5, edgecolor='white')
ax.text(2**-2, 2**6.25, "Peak BW %.0f GB/s" % (PEAK_BW),
        horizontalalignment='left', rotation=36.35,
        verticalalignment="bottom",
        bbox=props, zorder=2)

ax.text(2**-2.8, 2**3.8, "Triad BW %.0f GB/s" % (STREAM_TRIAD),
        horizontalalignment='left', rotation=36.35,
        verticalalignment="bottom",
        bbox=props, zorder=2)

ax.text(2**8, 2**10.2, "1 AVX mul + 1 AVX add/cycle %.0f Gflop/s" %
        PEAK_FLOPS,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=props, zorder=2)

ax.text(2**8, 2**9.2, "1 AVX op/cycle %.0f Gflop/s" %
        (PEAK_FLOPS / 2),
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=props, zorder=2)

ax.text(2**8, 2**6.4, "1 scalar op/cycle %.0f Gflop/s" %
        (PEAK_FLOPS / 8),
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=props, zorder=2)

ax.arrow(2**1, 2**3, 2**5 - 2**1, 0,
         head_width=1, fc="k", ec="k",
         linewidth=1.5, head_length=4, zorder=3,
         color="grey")

ax.text(2**2, 2**3.2, "Increasing degree",
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox=props, zorder=2)

markers = iter(["o", "s", "^", "v", ">", "<", "D", "p", "h", "*"])

colours = seaborn.color_palette(n_colors=5)

for dim in [2, 3]:
    colors = iter(colours)
    for typ in sorted(dataframe["type"].drop_duplicates()):
        sliced = (dataframe.loc[lambda df: df.type == typ]
                  .loc[lambda df: df.dimension == dim]
                  .loc[lambda df: df.num_processes == num_processes])
        grouped = sliced.groupby(["degree"], as_index=False)
        sliced = grouped.mean()

        name = {"aij": "AIJ",
                "matfree": "matrix-free",
                "nest": "Nest"}[typ]
        dimstr = {2: "2D",
                  3: "3D"}[dim]

        if typ != "matfree":
            ax.plot(sliced.assemble_ai, sliced.assemble_flops /
                    sliced.assemble_time*1000**-3,
                    label="Assemble %s [%s]" % (name, dimstr),
                    marker=next(markers),
                    markersize=10,
                    linestyle="",
                    zorder=3,
                    color=next(colors))

        ax.plot(sliced.matvec_ai, sliced.matmult_flops /
                sliced.matmult_time*1000**-3,
                label="MatMult %s [%s]" % (name, dimstr),
                marker=next(markers),
                markersize=10,
                linestyle="",
                zorder=3,
                color=next(colors))

for tick in ax.get_xticklabels():
    tick.set_fontsize(FONTSIZE)
for tick in ax.get_yticklabels():
    tick.set_fontsize(FONTSIZE)

handles, labels = ax.get_legend_handles_labels()

legend = fig.legend(handles, labels,
                    loc=9,
                    bbox_to_anchor=(0.5, 1.2),
                    bbox_transform=fig.transFigure,
                    ncol=2,
                    handlelength=4,
                    fontsize=FONTSIZE,
                    numpoints=1,
                    frameon=False)

seaborn.despine(fig)

fig.savefig("rb-roofline.pdf",
            orientation="landscape",
            format="pdf",
            transparent=True,
            bbox_inches="tight",
            bbox_extra_artists=[legend])
