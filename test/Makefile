falm = ..

include $(falm)/src/Makefile

cuflag = $(flag) --default-stream per-thread

heat: eql1
	$(nvnvcxx) -cuda -c heat1d.cpp $(flag) -o bin/heat1d.o
	$(nvnvcxx) -cuda $(mvl1obj) $(eql1obj) bin/heat1d.o -o bin/heat1d

heat2: eq
	$(nvmpicxx) -cuda -c heat1d2.cpp $(flag) -o bin/heat1d2.o
	$(nvmpicxx) -cuda $(mvobj) $(eqobj) $(cpmobj) bin/heat1d2.o -o bin/heat1d2

cpmtest2: cpm
	$(nvmpicxx) -cuda -c cpmtest2.cpp $(flag) -o bin/cpmtest2.o
	$(nvmpicxx) -cuda $(cpmobj) bin/cpmtest2.o -o bin/cpmtets2

cpmtestv2: cpm
	$(nvmpicxx) -cuda -c cpmtestv2.cpp $(flag) -o bin/cpmtestv2.o
	$(nvmpicxx) -cuda $(cpmobj) bin/cpmtestv2.o -o bin/cpmtestv2