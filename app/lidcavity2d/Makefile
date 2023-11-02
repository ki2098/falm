falmdir = ../..
include $(falmdir)/src/Makefile

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

lc2d: bc coord cfddevcall eqdevcall poisson
	$(nvnvcc) -c main.cpp $(flag) -o bin/lc2d.o
	$(nvnvcc) $(bcobj) $(coordobj) $(eqdevcallobj) $(mvdevcallobj) $(cfddevcallobj) $(poissonobj) bin/lc2d.o -o bin/lc2d