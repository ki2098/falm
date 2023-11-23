#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-16"
#PJM -L "vnode=2"
#PJM -L "vnode-core=36"
#PJM -L "elapse=01:00:00"
#PJM -j
#PJM -X
#PJM -o "turbine1.900x300x300.8.log"

module load nvhpc/nvhpc_20.11

date

mpirun -np 8 --map-by ppr:2:socket -display-devel-map -display-devel-map --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/t1 strong/large

# --mca btl_smcuda_use_cuda_ipc 0
# nvprof --log-file %q{OMPI_COMM_WORLD_RANK}.prof
# nvprof --log-file %q{OMPI_COMM_WORLD_RANK}.prof --track-memory-allocations on --print-gpu-trace

# tar cvzf data.tar.gz data
