nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc


flag   = --std=c++11

mvl1obj    = $(falm)/bin/MVL1.o
mvl2obj    = 
eql1obj    = $(falm)/bin/structEqL1.o
eql2obj    = $(falm)/bin/structEqL2.o
cpml1obj   = $(falm)/bin/CPML1.o
cpml2obj   = 
cfdl1obj   = $(falm)/bin/CFDL1.o
cfdl2obj   = 

mvobj      = $(mvl1obj) $(mvl2obj)
eqobj      = $(eql1obj) $(eql2obj)
cpmobj     = $(cpml1obj) $(cpml2obj)
cfdobj     = $(cfdl1obj) $(cfdl2obj)
falmobj    = $(mvobj) $(eqobj) $(cpmobj) $(cfdobj)

cpml1:
	$(nvnvcc) -c $(falm)/src/dev/CPML1v2.cu $(cuflag) -o $(cpml1obj)

cpml2: cpml1

cpm: cpml2

mvl1:
	$(nvnvcc) -c $(falm)/src/dev/MVL1.cu $(cuflag) -o $(mvl1obj)

mvl2: mvl1 cpm

mv: mvl2

eql1: mvl1
	$(nvnvcc) -c $(falm)/src/dev/structEqL1.cu $(cuflag) -o $(eql1obj)

eql2: eql1 cpm mvl2
	$(nvmpicxx) -cuda -c $(falm)/src/structEqL2.cpp $(flag) -o $(eql2obj)

eq: eql2

cfdl1:
	$(nvnvcc) -c $(falm)/src/dev/FalmCFDL1.cu $(cuflag) -o $(cfdl1obj)