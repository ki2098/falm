nvnvcc = /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/cuda/bin/nvcc
cudaflag = --default-stream per-thread

bcobj = bin/bc.o
bc:
	$(nvnvcc) -c boundaryCondition.cu $(cudaflag) --std=c++11 -o $(bcobj)

coordobj = bin/coord.o
coord:
	$(nvnvcc) -c coordinate.cu $(cudaflag) --std=c++11 -o $(coordobj)

lc2d: bc coord
	$(nvnvcc) -c main.cpp --std=c++11 -o bin/lc2d.o
	$(nvnvcc) $(bcobj) $(coordobj) bin/lc2d.o -o bin/lc2d