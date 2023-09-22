nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc

mvl1:
	$(nvnvcc) -c ../src/dev/MVL1.cu --std=c++11 -o bin/MVL1.o

eql1:
	$(nvnvcc) -c ../src/dev/structEqL1.cu --std=c++11 -o bin/structEqL1.o

cpml1:
	$(nvnvcc) -c ../src/dev/CPML1.cu --std=c++11 -o bin/CPML1.o

dot: mvl1
	$(nvnvcxx) -cuda -c dot.cpp --std=c++11 -o bin/dot.o
	$(nvnvcxx) -cuda bin/MVL1.o bin/dot.o -o bin/dot

heat: mvl1 eql1
	$(nvnvcxx) -cuda -c heat1d.cpp --std=c++11 -o bin/heat1d.o
	$(nvnvcxx) -cuda bin/MVL1.o bin/heat1d.o bin/structEqL1.o -o bin/heat1d

assign:
	$(nvnvcxx) -cuda assign.cpp --std=c++11 -o bin/assign

cpml2: cpml1
	$(nvmpicxx) -cuda -c ../src/CPML2.cpp --std=c++11 -o bin/CPML2.o

cpmtest: cpml2 
	$(nvmpicxx) -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	$(nvmpicxx) -cuda bin/CPML2.o bin/cpmtest.o bin/CPML1.o -o bin/cpmtest

cpml2ito:
	mpic++ -cuda -c ../src/CPML2.cpp --std=c++11 -o bin/CPML2.o

cpml1ito:
	nvcc -c ../src/dev/CPML1.cu --std=c++11 -o bin/CPML1.o

cpmtestito: cpml1ito cpml2ito
	mpic++ -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	mpic++ -cuda bin/CPML1.o bin/cpmtest.o bin/CPML2.o -o bin/cpmtest

cpmtest2: 
	$(nvmpicxx) -cuda cpmtest2.cpp -o bin/cpmtest2