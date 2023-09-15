mvbasic:
	nvcc -c ../src/dev/mvbasic.cu -o bin/mvbasic.o

mveq:
	nvcc -c ../src/dev/structMVEq.cu -o bin/structMVEq.o

dot: mvbasic
	nvc++ -cuda -c dot.cpp -o bin/dot.o
	nvc++ -cuda bin/mvbasic.o bin/dot.o -o bin/dot

heat: mvbasic mveq
	nvc++ -cuda -c heat1d.cpp -o bin/heat1d.o
	nvc++ -cuda bin/mvbasic.o bin/heat1d.o bin/structMVEq.o -o bin/heat1d