falm = ../..
include $(falm)/src/Makefile

cuflag = $(flag) --default-stream per-thread

bcobj = bin/bc.o
bc:
	$(nvnvcc) -c boundaryCondition.cu $(cuflag) -o $(bcobj)

coordobj = bin/coord.o
coord:
	$(nvnvcc) -c coordinate.cu $(cuflag) -o $(coordobj)

poissonobj = bin/poisson.o
poisson:
	$(nvnvcc) -c poisson.cu $(cuflag) -o $(poissonobj)

lc2d: bc coord cfdl1 eql1 poisson
	$(nvnvcc) -c main.cpp $(flag) -o bin/lc2d.o
	$(nvnvcc) $(bcobj) $(coordobj) $(eql1obj) $(mvl1obj) $(cfdl1obj) $(poissonobj) bin/lc2d.o -o bin/lc2d