nvmpicxx = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/comm_libs/mpi/bin/mpic++

mvbasic:
	nvcc -c ../src/dev/mvbasic.cu --std=c++11 -o bin/mvbasic.o

mveq:
	nvcc -c ../src/dev/structMVEq.cu --std=c++11 -o bin/structMVEq.o

cpmdev:
	nvcc -c ../src/dev/CPMDev.cu --std=c++11 -o bin/CPMDev.o

dot: mvbasic
	nvc++ -cuda -c dot.cpp --std=c++11 -o bin/dot.o
	nvc++ -cuda bin/mvbasic.o bin/dot.o -o bin/dot

heat: mvbasic mveq
	nvc++ -cuda -c heat1d.cpp --std=c++11 -o bin/heat1d.o
	nvc++ -cuda bin/mvbasic.o bin/heat1d.o bin/structMVEq.o -o bin/heat1d

assign:
	nvc++ -cuda assign.cpp --std=c++11 -o bin/assign

cpm: cpmdev
	$(nvmpicxx) -Mcuda=cuda12.1 -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	$(nvmpicxx) -Mcuda=cuda12.1 bin/cpmtest.o bin/CPMDev.o -o bin/cpm

cpm_ito: cpmdev
	mpic++ -cuda -gpu=cc60,cuda11.0,pinned -c cpmtest.cpp --std=c++11 -o bin/cpmtest.o
	mpic++ -cuda -gpu=cc60,cuda11.0,pinned bin/cpmtest.o bin/CPMDev.o -o bin/cpm