import pandas

data = pandas.read_csv("rb-weak-scale.csv")
data = data.sort_values("num_processes")

table = r"""DoFs ($\times 10^{6}$) & MPI processes & Newton its & Krylov its & Time to solution (s)\\
\hline
"""

lformat = r"""{dofs:.4g} & {num_processes:d} & {nits:d} & {kits:d} & {time:.3g}\\
"""

for np in data.num_processes:
    sliced = data.loc[lambda x: x.num_processes == np]
    table += lformat.format(dofs=(sliced.dofs.values[0] / 1e6),
                            num_processes=np,
                            nits=sliced.snes_its.values[0],
                            kits=sliced.ksp_its.values[0],
                            time=sliced.SNESSolve.values[0])

print(table)
