almdevcallobj = $(falmdir)/bin/AlmDevCall.o
almobj = 
alm_obj = $(almdevcallobj) $(almobj)

almdevcall:
	$(nvnvcc) -c $(falmdir)/src/alm/almDevCall.cu $(cuflag) -o $(almdevcallobj)

alm: almdevcall cpm