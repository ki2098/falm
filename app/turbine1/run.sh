#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-16"
#PJM -L "vnode=2"
#PJM -L "vnode-core=36"
#PJM -L "elapse=02:00:00"
#PJM -j
#PJM -X
#PJM -o "turbine1.8.log"

module load nvhpc/nvhpc_20.11

date

mpirun -np 8 --map-by ppr:2:socket -display-devel-map -display-devel-map --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} nvprof --log-file %q{OMPI_COMM_WORLD_RANK}.prof ./bin/t1

#  --mca btl_smcuda_use_cuda_ipc 0

tar cvzf data.tar.gz data
