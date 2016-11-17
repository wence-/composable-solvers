# flake8: noqa: F403
from argparse import ArgumentParser
import importlib, sys
import os
import pandas
from matplotlib import pyplot
pyplot.style.use('ggplot')

parser = ArgumentParser(description="""Plot matvec results""", add_help=False)

parser.add_argument("--problem", choices=["poisson",
                                          "elasticity",
                                          "navier_stokes",
                                          "rayleigh_benard"],
                    help="Which problem to get data from")

parser.add_argument("--results-directory",
                    help="Where to get results from")

parser.add_argument("--help", action="store_true",
                    help="Show help")


parser.add_argument("--output-directory",
                    help="Where to put plots")

parser.add_argument("--num-processes", default=1, type=int,
                    help="Number of processes data was created on")

args, _ = parser.parse_known_args()


if args.help:
    parser.print_help()


if args.problem is None:
    print "Must provide problem type\n"
    sys.exit(1)


if args.output_directory is None:
    print "Must provide output directory\n"
    sys.exit(1)


if args.results_directory is None:
    print "Must provide results directory\n"
    sys.exit(1)


module = importlib.import_module("problem.%s" % args.problem)
problem = module.Problem()

problem.nprocs = args.num_processes

results = os.path.join(os.path.abspath(args.output_directory),
                       problem.datastore_name)

store = pandas.HDFStore(results, mode="r")

typs = sorted(store.keys())

names = {"/aij": "Assembled AIJ",
         "/nest": "Assembled Nest",
         "/matfree": "Matrix-free"}

title = "%s dimension %d, size %d, degree %d" % (problem.name, problem.dimension,
                                                 problem.N, problem.degree)

ax = store[typs[0]].plot(y="time", label=names[typs[0]])

for typ in typs[1:]:
    store[typ].plot(y="time", label=names[typ], ax=ax)

ax.set_xlabel("MatVecs")
ax.set_ylabel("Time (s)")
ax.set_title(title)

output_name = os.path.join(os.path.abspath(args.output_directory),
                           os.path.splitext(problem.datastore_name)[0] + ".pdf")
pyplot.savefig(output_name, format="pdf")

store.close()
