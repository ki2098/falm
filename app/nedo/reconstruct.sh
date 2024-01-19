#!/usr/bin/bash
#PJM -L "rscunit=ito-a"
#PJM -L "rscgrp=ito-ss"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=05:00:00"
#PJM -j
#PJM -X
#PJM -o "reconstruct.log"

. ./env

${falmdir}/bin/reconstructor data/uvwp