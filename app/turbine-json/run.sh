#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-16"
#PJM -L "vnode=2"
#PJM -L "vnode-core=36"
#PJM -L "elapse=01:00:00"
#PJM -j
#PJM -X
#PJM -o "turbine.8.log"

module load nvhpc/nvhpc_20.11

date

mpirun -n 8 --map-by ppr:2:socket:PE=9 --bind-to core --report-bindings --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/main