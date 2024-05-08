import numpy as np
import sys

path = sys.argv[1]

imax = np.fromfile(path, dtype='uint64', count=1)
jmax = np.fromfile(path, dtype='uint64', count=1)
kmax = np.fromfile(path, dtype='uint64', count=1)
nvar = np.fromfile(path, dtype='uint64', count=1)
gc   = np.fromfile(path, dtype='uint64', count=1)
step = np.fromfile(path, dtype='uint64', count=1)
time = np.fromfile(path, dtype='float64', count=1)
dtype = np.fromfile(path, dtype='uint64', count=1)

print("%s info:"%(path))
print("size=(%d %d %d %d)"%(imax, jmax, kmax, nvar))
print("gc=%d"%(gc))
print("step=%d"%(step))
print("time=%s"%(time))
print("data width=%d"%(dtype))