falmdir = ../..
include $(falmdir)/src/Makefile

cuflag = $(flag) --default-stream per-thread

main: Falm
	$(nvmpicxx) -cuda -c main.cpp $(flag) -o bin/main.o
	$(nvmpicxx) -cuda $(falmobj) bin/main.o -o bin/main