nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc


flag   = --std=c++11

mvl1obj    = $(falmdir)/bin/MVL1.o
mvl2obj    = 
eql1obj    = $(falmdir)/bin/structEqL1.o
eql2obj    = $(falmdir)/bin/structEqL2.o
cpml1obj   = $(falmdir)/bin/CPML1.o
cpml2obj   = 
cfdl1obj   = $(falmdir)/bin/CFDL1.o
cfdl2obj   = $(falmdir)/bin/CFDL2.o

mvobj      = $(mvl1obj) $(mvl2obj)
eqobj      = $(eql1obj) $(eql2obj)
cpmobj     = $(cpml1obj) $(cpml2obj)
cfdobj     = $(cfdl1obj) $(cfdl2obj)
falmobj    = $(mvobj) $(eqobj) $(cpmobj) $(cfdobj)

cpml1:
	$(nvnvcc) -c $(falmdir)/src/dev/CPML1v2.cu $(cuflag) -o $(cpml1obj)

cpml2: cpml1

cpm: cpml2

mvl1:
	$(nvnvcc) -c $(falmdir)/src/dev/MVL1.cu $(cuflag) -o $(mvl1obj)

mvl2: mvl1 cpm

mv: mvl2

eql1: mvl1
	$(nvnvcc) -c $(falmdir)/src/dev/structEqL1.cu $(cuflag) -o $(eql1obj)

eql2: eql1 cpm mvl2
	$(nvmpicxx) -cuda -c $(falmdir)/src/structEqL2.cpp $(flag) -o $(eql2obj)

eq: eql2

cfdl1:
	$(nvnvcc) -c $(falmdir)/src/dev/FalmCFDL1.cu $(cuflag) -o $(cfdl1obj)

cfdl2: cfdl1 cpm
	$(nvmpicxx) -cuda -c $(falmdir)/src/FalmCFDL2.cpp $(flag) -o $(cfdl2obj)

cfd: cfdl2

Falm: cfd eq

