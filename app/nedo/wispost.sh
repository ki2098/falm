#!/usr/bin/bash

#PJM -g gg18
#PJM -o nedopost.log
#PJM -j
#PJM -X
#PJM -L rscgrp=prepost
#PJM -L node=1
#PJM -L elapse=02:00:00

. ./env

ls -l --block-size=M data

date
${falmdir}/bin/reconstructor data/uvwp
${falmdir}/bin/visifalm data/uvwp
date