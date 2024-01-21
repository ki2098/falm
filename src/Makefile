ifeq ($(platform),ito)
	nvmpicxx = mpic++
	nvnvcxx = nvc++
	nvnvcc = nvcc
	c++ = icpc
	fomp = -qopenmp
	fstl =
else
	nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
	nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
	nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc
	c++ = g++
	fomp = -fopenmp
	fstl =
endif


flag   = -O3 --std=c++11

include $(falmdir)/src/rmcp/Makefile

mvdevcallobj    = $(falmdir)/bin/MVDevCall.o
eqdevcallobj    = $(falmdir)/bin/FalmEqDevCall.o
cpmdevcallobj   = $(falmdir)/bin/CPMDevCall.o
cfddevcallobj   = $(falmdir)/bin/CFDDevCall.o

mvobj           = 
eqobj           = $(falmdir)/bin/FalmEq.o
cpmobj          = 
cfdobj          = $(falmdir)/bin/CFD.o

alm_obj         = $(rmcp_obj)

mv_obj      = $(mvdevcallobj) $(mvobj)
eq_obj      = $(eqdevcallobj) $(eqobj)
cpm_obj     = $(cpmdevcallobj) $(cpmobj)
cfd_obj     = $(cfddevcallobj) $(cfdobj)
falmobj    = $(mv_obj) $(eq_obj) $(cpm_obj) $(cfd_obj) $(alm_obj)

alm: rmcp

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


Falm: cfd eq alm

reconstructor:
	$(c++) $(falmdir)/src/postfalm/reconstructor.cpp -std=c++11 $(fomp) $(fstl) -o $(falmdir)/bin/reconstructor

visifalm:
	$(c++) $(falmdir)/src/postfalm/visifalm.cpp -std=c++11 $(fstl) -o $(falmdir)/bin/visifalm

postfalm: reconstructor visifalm

showenv:
	@echo nvmpicxx = $(nvmpicxx)
	@echo nvnvcc = $(nvnvcc)  
	@echo nvnvcxx = $(nvnvcxx) 