#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=01:00:00"
#PJM -j
#PJM -X
#PJM -o "turbine1.weak.2.log"

module load nvhpc/nvhpc_20.11

date

mpirun -np 2 --map-by ppr:1:socket --bind-to core --report-bindings --mca plm_rsh_agent /bin/pjrsh -machinefile ${PJM_O_NODEINF} ./bin/t1 weak/2

# --mca mpi_leave_pinned 0
# --mca btl_smcuda_use_cuda_ipc 0
# nvprof --log-file %q{OMPI_COMM_WORLD_RANK}.prof
# nvprof --log-file %q{OMPI_COMM_WORLD_RANK}.prof --track-memory-allocations on --print-gpu-trace

# tar cvzf data.tar.gz data
