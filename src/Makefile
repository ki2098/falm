nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc


flag   = --std=c++11

mvdevcallobj    = $(falmdir)/bin/MVDevCall.o
mvobj    = 
eqdevcallobj    = $(falmdir)/bin/FalmEqDevCall.o
eqobj    = $(falmdir)/bin/FalmEq.o
cpmdevcallobj   = $(falmdir)/bin/CPMDevCall.o
cpmcommobj   = 
cfddevcallobj   = $(falmdir)/bin/CFDDevCall.o
cfdobj   = $(falmdir)/bin/CFD.o

mv_obj      = $(mvdevcallobj) $(mvobj)
eq_obj      = $(eqdevcallobj) $(eqobj)
cpm_obj     = $(cpmdevcallobj) $(cpmcommobj)
cfd_obj     = $(cfddevcallobj) $(cfdobj)
falmobj    = $(mv_obj) $(eq_obj) $(cpm_obj) $(cfd_obj)

cpmdevcall:
	$(nvnvcc) -c $(falmdir)/src/dev/CPMDevCall.cu $(cuflag) -o $(cpmdevcallobj)

cpmcomm: cpmdevcall

cpm: cpmcomm

mvdevcall:
	$(nvnvcc) -c $(falmdir)/src/dev/MVDevCall.cu $(cuflag) -o $(mvdevcallobj)

mv: mvdevcall cpm

eqdevcall: mvdevcall
	$(nvnvcc) -c $(falmdir)/src/dev/FalmEqDevCall.cu $(cuflag) -o $(eqdevcallobj)

eq: eqdevcall cpm mv
	$(nvmpicxx) -cuda -c $(falmdir)/src/FalmEq.cpp $(flag) -o $(eqobj)

cfddevcall:
	$(nvnvcc) -c $(falmdir)/src/dev/FalmCFDDevCall.cu $(cuflag) -o $(cfddevcallobj)

cfd: cfddevcall cpm
	$(nvmpicxx) -cuda -c $(falmdir)/src/FalmCFD.cpp $(flag) -o $(cfdobj)


Falm: cfd eq

