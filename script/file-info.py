import numpy as np
import sys

path = sys.argv[1]

imax = np.fromfile(path, dtype='uint64', count=1, offset=0)
jmax = np.fromfile(path, dtype='uint64', count=1, offset=8)
kmax = np.fromfile(path, dtype='uint64', count=1, offset=16)
nvar = np.fromfile(path, dtype='uint64', count=1, offset=24)
gc   = np.fromfile(path, dtype='uint64', count=1, offset=32)
step = np.fromfile(path, dtype='uint64', count=1, offset=40)
time = np.fromfile(path, dtype='float64', count=1, offset=48)
dtype = np.fromfile(path, dtype='uint64', count=1, offset=56)

print("%s info:"%(path))
print("size=(%d %d %d %d)"%(imax, jmax, kmax, nvar))
print("gc=%d"%(gc))
print("step=%d"%(step))
print("time=%s"%(time))
print("data width=%d"%(dtype))