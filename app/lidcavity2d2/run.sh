#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=02:00:00"
#PJM -j
#PJM -X
#PJM -o "cavity++.log"

module load nvhpc/nvhpc_20.11

date

mpirun -np 4 bin/lc2d2 2 2