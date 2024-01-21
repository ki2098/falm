#!/usr/bin/bash
#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-4"
#PJM -L "vnode=1"
#PJM -L "vnode-core=36"
#PJM -L "elapse=05:00:00"
#PJM -j
#PJM -X
#PJM -o "reconstruct.log"

. ./env

${falmdir}/bin/reconstructor data/uvwp
${falmdir}/bin/visifalm data/uvwp