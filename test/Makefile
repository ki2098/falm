nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc

cudaflag = 

mvl1obj    = bin/MVL1.o
mvl2obj    = 
eql1obj    = bin/structEqL1.o
eql2obj    = bin/structEqL2.o
cpml1obj   = bin/CPML1.o
cpml2obj   = bin/CPML2.o
cpml1v2obj = bin/CPML1v2.o
cpml2v2obj = 

mvobj      = $(mvl1obj) $(mvl2obj)
eqobj      = $(eql1obj) $(eql2obj)
cpmobj     = $(cpml1v2obj) $(cpml2v2obj)

mvl1:
	$(nvnvcc) -c ../src/dev/MVL1.cu $(cudaflag) --std=c++11 -o $(mvl1obj)

mvl1ito:
	nvcc -c ../src/dev/MVL1.cu $(cudaflag) --std=c++11 -o $(mvl1obj)

eql1:
	$(nvnvcc) -c ../src/dev/structEqL1.cu $(cudaflag) --std=c++11 -o $(eql1obj)

eql1ito:
	nvcc -c ../src/dev/structEqL1.cu $(cudaflag) --std=c++11 -o $(eql1obj)

cpml1:
	$(nvnvcc) -c ../src/dev/CPML1.cu $(cudaflag) --std=c++11 -o $(cpml1obj)

cpml1v2:
	$(nvnvcc) -c ../src/dev/CPML1v2.cu $(cudaflag) --std=c++11 -o $(cpml1v2obj)

cpml1ito:
	nvcc -c ../src/dev/CPML1.cu $(cudaflag) --std=c++11 -o $(cpml1obj)

cpml1v2ito:
	nvcc -c ../src/dev/CPML1v2.cu $(cudaflag) --std=c++11 -o $(cpml1v2obj)

cpml2: cpml1
	$(nvmpicxx) -cuda -c ../src/CPML2.cpp --std=c++11 -o $(cpml2obj)

cpml2v2: cpml1v2

cpml2ito: cpml1ito
	mpic++ -cuda -c ../src/CPML2.cpp --std=c++11 -o $(cpml2obj)

cpml2v2ito: cpml1v2ito

cpm: cpml2v2

cpmito: cpml2v2ito

dot: mvl1
	$(nvnvcxx) -cuda -c dot.cpp --std=c++11 -o bin/dot.o
	$(nvnvcxx) -cuda $(mvl1obj) bin/dot.o -o bin/dot

heat: mvl1 eql1
	$(nvnvcxx) -cuda -c heat1d.cpp --std=c++11 -o bin/heat1d.o
	$(nvnvcxx) -cuda $(mvl1obj) $(eql1obj) bin/heat1d.o -o bin/heat1d

assign:
	$(nvnvcxx) -cuda assign.cpp --std=c++11 -o bin/assign

cpmtest: cpml2 
	$(nvmpicxx) -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	$(nvmpicxx) -cuda $(cpml1obj) $(cpml2obj) bin/cpmtest.o -o bin/cpmtest

cpmtestv2: cpm
	$(nvmpicxx) -cuda -c cpmtestv2.cpp --std=c++11 -o bin/cpmtestv2.o
	$(nvmpicxx) -cuda $(cpmobj) bin/cpmtestv2.o -o bin/cpmtestv2

cpmtestito: cpml2ito
	mpic++ -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	mpic++ -cuda $(cpml1obj) $(cpml2obj) bin/cpmtest.o -o bin/cpmtest

cpmtestv2ito: cpmito
	mpic++ -cuda -c cpmtestv2.cpp --std=c++11 -o bin/cpmtestv2.o
	mpic++ -cuda $(cpmobj) bin/cpmtestv2.o -o bin/cpmtestv2

cpmtest2: 
	$(nvmpicxx) -cuda cpmtest2.cpp -o bin/cpmtest2

mvl2: mvl1 cpm

mvl2ito: mvl1ito cpmito

eql2: eql1 mvl2 cpm
	$(nvmpicxx) -cuda -c ../src/structEqL2.cpp --std=c++11 -o bin/structEqL2.o

eql2ito: eql1ito mvl2ito cpmito
	mpic++ -cuda -c ../src/structEqL2.cpp --std=c++11 -o bin/structEqL2.o

heat2: eql2
	$(nvmpicxx) -cuda -c heat1d2.cpp --std=c++11 -o bin/heat1d2.o
	$(nvmpicxx) -cuda $(mvobj) $(eqobj) $(cpmobj) bin/heat1d2.o -o bin/heat1d2

heat2ito: eql2ito
	mpic++ -cuda -c heat1d2.cpp --std=c++11 -o bin/heat1d2.o
	mpic++ -cuda $(mvobj) $(eqobj) $(cpmobj) bin/heat1d2.o -o bin/heat1d2