#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=12:00:00"
#PJM -j
#PJM -X
#PJM -o "nedo.log"

date
. ./env

module load nvhpc/nvhpc_20.11

mpirun -n 4 --map-by ppr:2:socket:PE=9 --bind-to core --report-bindings --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/main

module unload nvhpc/nvhpc_20.11

ls -l --block-size=M data

module load oneapi/2022.3.1

${falmdir}/bin/reconstructor data/uvwp
${falmdir}/bin/visifalm data/uvwp