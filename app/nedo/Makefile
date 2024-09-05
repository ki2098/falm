falmdir = ../..
include $(falmdir)/src/Makefile

cuflag = $(flag) --default-stream per-thread

bcobj = bin/bc.o
bc:
	$(nvnvcc) -c bcdevcall.cu $(cuflag) -o $(bcobj)

drydist: Falm
	$(nvmpicxx) -cuda -c drydist.cpp $(flag) -o bin/drydist.o
	$(nvmpicxx) -cuda $(cppstd) -lstdc++ $(falmobj) bin/drydist.o -o bin/drydist

main: Falm bc
	$(nvmpicxx) -cuda -c main.cpp $(flag) -o bin/main.o
	$(nvmpicxx) -cuda $(cppstd) -lstdc++ $(falmobj) $(bcobj) bin/main.o -o bin/main