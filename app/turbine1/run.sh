#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=02:00:00"
#PJM -j
#PJM -X
#PJM -o "turbine1.4.log"

module load nvhpc/nvhpc_20.11

date

mpirun -np 4 --map-by ppr:2:socket -display-devel-map -display-devel-map --mca btl_smcuda_use_cuda_ipc 0 --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/t1

tar cvzf data.tar.gz data
