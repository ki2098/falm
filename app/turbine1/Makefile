falmdir = ../..
include $(falmdir)/src/Makefile

cuflag = $(flag) --default-stream per-thread

bcobj = bin/bc.o
bc:
	$(nvnvcc) -c bcdev.cu $(cuflag) -o $(bcobj)

main: Falm bc
	$(nvmpicxx) -cuda -c main.cpp $(flag) -o bin/main.o
	$(nvmpicxx) -cuda $(falmobj) $(bcobj) bin/main.o -o bin/t1