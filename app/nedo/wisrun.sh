#!/usr/bin/bash

#PJM -g gg18
#PJM -o nedo.log
#PJM -j
#PJM -X
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=03:00:00
#PJM --mpi proc=8

date
. ./env

module load nvidia
module load nvmpi
module list

cp wissetup.json setup.json

mpiexec -machinefile $PJM_O_NODEINF -n $PJM_MPI_PROC -npernode 8 --report-bindings bin/main

date