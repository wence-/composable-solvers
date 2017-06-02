import pandas

data_hypre = pandas.read_csv("poisson-weak-hypre.csv")
data_schwarz = pandas.read_csv("poisson-weak-schwarz_rich.csv")

data_hypre = data_hypre.sort_values("num_processes")
data_schwarz = data_schwarz.sort_values("num_processes")

table = r"""DoFs ($\times 10^{6}$) & MPI processes & \multicolumn{2}{|c|}{Krylov its} & \multicolumn{2}{|c}{Time to solution (s)}\\
 & & hypre & schwarz & hypre & schwarz\\
\hline
"""

lformat = r"""{dofs:.4g} & {num_processes:d} & {hits:d} & {sits:d} & {htime:.3g} & {stime:.3g}\\
"""

for np in data_hypre.num_processes:
    hypre = data_hypre.loc[lambda x: x.num_processes == np]
    schwarz = data_schwarz.loc[lambda x: x.num_processes == np]
    table += lformat.format(dofs=(hypre.dofs.values[0] / 1e6),
                            num_processes=np,
                            hits=hypre.ksp_its.values[0],
                            sits=schwarz.ksp_its.values[0],
                            htime=hypre.KSPSolve.values[0],
                            stime=schwarz.KSPSolve.values[0])

print table
