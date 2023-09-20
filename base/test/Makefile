nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++
nvnvcxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvc++
nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc

mvbasic:
	$(nvnvcc) -c ../src/dev/mvbasic.cu --std=c++11 -o bin/mvbasic.o

mveq:
	$(nvnvcc) -c -c ../src/dev/structMVEq.cu --std=c++11 -o bin/structMVEq.o

cpmdev:
	$(nvnvcc) -c -c ../src/dev/CPMDev.cu --std=c++11 -o bin/CPMDev.o

dot: mvbasic
	$(nvnvcxx) -cuda -c dot.cpp --std=c++11 -o bin/dot.o
	$(nvnvcxx) -cuda bin/mvbasic.o bin/dot.o -o bin/dot

heat: mvbasic mveq
	$(nvnvcxx) -cuda -c heat1d.cpp --std=c++11 -o bin/heat1d.o
	$(nvnvcxx) -cuda bin/mvbasic.o bin/heat1d.o bin/structMVEq.o -o bin/heat1d

assign:
	$(nvnvcxx) -cuda assign.cpp --std=c++11 -o bin/assign

cpm: cpmdev
	$(nvmpicxx) -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	$(nvmpicxx) -cuda bin/cpmtest.o bin/CPMDev.o -o bin/cpm

cpm_ito: cpmdev
	mpic++ -cuda -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	mpic++ -cuda bin/cpmtest.o bin/CPMDev.o -o bin/cpm