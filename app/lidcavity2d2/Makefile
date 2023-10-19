falmdir = ../..
include $(falmdir)/src/Makefile

cuflag = $(flag) --default-stream per-thread

bcobj = bin/bc.o
bc:
	$(nvnvcc) -c boundaryDev.cu $(cuflag) -o $(bcobj)

coordobj = bin/coord.o
coord:
	$(nvnvcc) -c coordinate.cu $(cuflag) -o $(coordobj)

poissonobj = bin/poisson.o
poisson:
	$(nvnvcc) -c poissonDev.cu $(cuflag) -o $(poissonobj)

lc2d2: bc coord poisson Falm
	$(nvmpicxx) -cuda -c main.cpp $(flag) -o bin/main.o
	$(nvmpicxx) -cuda $(bcobj) $(coordobj) $(poissonobj) $(falmobj) bin/main.o -o bin/lc2d2