
rmcpdevcallobj = $(falmdir)/bin/RmcpDevCall.o
rmcpobj        = 
rmcp_obj       = $(rmcpdevcallobj) $(rmcpobj)

rmcpdevcall:
	$(nvnvcc) -c $(falmdir)/src/rmcp/almDevCall.cu $(cuflag) -o $(rmcpdevcallobj)

rmcp: rmcpdevcall cpm