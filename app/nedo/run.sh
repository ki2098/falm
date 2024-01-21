#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=12:00:00"
#PJM -j
#PJM -X
#PJM -o "nedo.log"

module load nvhpc/nvhpc_20.11
module load oneapi/2022.3.1

date

. ./env

mpirun -n 4 --map-by ppr:2:socket:PE=9 --bind-to core --report-bindings --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/main

ls -l --block-size=M data

${falmdir}/bin/reconstructor data/uvwp
${falmdir}/bin/visifalm data/uvwp