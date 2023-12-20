falmdir = ../..
include $(falmdir)/src/Makefile

cuflag = $(flag) --default-stream per-thread

bcobj = 
bc:
	$(nvnvcc) -c bcdev.cu $(cuflag) -o $(bcobj)

main: Falm 
	$(nvmpicxx) -cuda -c main.cpp $(flag) -o bin/main.o
	$(nvmpicxx) -cuda $(falmobj) $(bcobj) bin/main.o -o bin/main