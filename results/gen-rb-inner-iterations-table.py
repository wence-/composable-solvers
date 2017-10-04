import pandas
import pickle
import sys

if sys.version_info[0] > 2:
    loads = lambda s: pickle.loads(s.encode(), encoding="bytes")
else:
    loads = pickle.loads

data = pandas.read_csv("rb-weak-scale.csv")
data = data.sort_values("num_processes")

table = r"""DoFs ($\times 10^{6}$) & Navier-Stokes iterations & Temperature iterations\\
\hline
"""

lformat = r"""{dofs:.4g} & {nits:d} ({anits:.3g}) & {tits:d} ({atits:.2g})\\
"""

for np in data.num_processes:
    sliced = data.loc[lambda x: x.num_processes == np]
    nits = loads(sliced.fs0_its.values[0])
    tits = loads(sliced.fs1_its.values[0])
    table += lformat.format(dofs=(sliced.dofs.values[0] / 1e6),
                            nits=nits.sum(),
                            anits=nits.mean(),
                            tits=tits.sum(),
                            atits=tits.mean())

print(table)
