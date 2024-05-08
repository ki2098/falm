import numpy as np
import sys

file=open(sys.argv[1], 'rb')

imax = np.fromfile(file, dtype='uint64', count=1, offset=0)
jmax = np.fromfile(file, dtype='uint64', count=1, offset=8)
kmax = np.fromfile(file, dtype='uint64', count=1, offset=16)
nvar = np.fromfile(file, dtype='uint64', count=1, offset=24)
gc   = np.fromfile(file, dtype='uint64', count=1, offset=32)
step = np.fromfile(file, dtype='uint64', count=1, offset=40)
time = np.fromfile(file, dtype='float64', count=1, offset=48)
dtype = np.fromfile(file, dtype='uint64', count=1, offset=56)

file.close()

print("%s info:"%(sys.argv[1]))
print("size=(%d %d %d %d)"%(imax, jmax, kmax, nvar))
print("gc=%d"%(gc))
print("step=%d"%(step))
print("time=%s"%(time))
print("data width=%d"%(dtype))