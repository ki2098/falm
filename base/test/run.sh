#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-16-dbg"
#PJM -L "vnode=4"
#PJM -L "vnode-core=36"
#PJM -L "elapse=00:05:00"
#PJM -j
#PJM -X
#PJM -o "heat++.log"

module load nvhpc/nvhpc_20.11

date

# mpirun -np 16 bin/cpmtestv2 2 4 2
mpirun -np 16 --bind-to core --map-by ppr:2:socket --rank-by socket:span  --report-bindings --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} bin/heat1d2