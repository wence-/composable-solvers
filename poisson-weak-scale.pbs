#!/bin/bash --login

#PBS -N Poisson-weak

module swap PrgEnv-cray PrgEnv-gnu
module load python-compute
module load cray-libsci
module load cmake
module load cray-hdf5-parallel

export CRAYPE_LINK_TYPE=dynamic

export PETSC_DIR=/work/n02/n02/lmn02/petsc-gnu51-ivybridge-int64

unset PYTHONPATH
unalias pip
export PIP_CERT=/home/y07/y07/cse/curl/7.34.0/lib/ca-bundle.crt
export HDF5_VERSION=1.8.14

BASE=/work/n02/n02/lmn02

export CC=cc
export CXX=CC
export FC=ftn
export F90=$FC
export F77=$FC
export MPICC=cc
export MPICXX=CC
export MPIFC=ftn
export MPIF90=$MPIFC
export MPIF77=$MPIFC

export MPICH_GNI_FORK_MODE=FULLCOPY
export PYOP2_CACHE_DIR=$BASE/pyop2-cache
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=$BASE/pyop2-cache

export WORK=/work/n02/n02/lmn02

. $BASE/firedrake-int64/bin/activate

export PYTHONPATH=$WORK/composable-solvers:$PYTHONPATH

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from
# (remember this should be on the /work filesystem)
cd $PBS_O_WORKDIR

# Set the number of threads to 1
#   This prevents any system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1


NPROC=${NUM_PES}

aprun -n ${NPROC} -N 24 python poisson-new-weak.py --dimension 3 --degree 4 \
      --parameters hypre \
      --results-file poisson-weak-hypre.csv \
      -log_view poisson-weak-hypre-${NPROC}.log > poisson-weak-hypre-${NPROC}.out

aprun -n ${NPROC} -N 24 python poisson-new-weak.py --dimension 3 --degree 4 \
      --parameters schwarz_rich \
      --results-file poisson-weak-schwarz_rich.csv \
      -log_view poisson-weak-schwarz_rich-${NPROC}.log > poisson-weak-schwarz_rich-${NPROC}.out
