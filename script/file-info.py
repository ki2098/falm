import numpy as np
import sys

# print(np.__version__)

path = sys.argv[1]
file = open(path, 'rb')

imax = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
jmax = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
kmax = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
nvar = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
gc   = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
step = np.fromfile(file, dtype='uint64', count=1)
# file.seek(8, 1)
time = np.fromfile(file, dtype='float64', count=1)
# file.seek(8, 1)
dtype = np.fromfile(file, dtype='uint64', count=1)

print("%s info:"%(path))
print("size=(%d %d %d)"%(imax, jmax, kmax))
print("number of vars=%d"%(nvar))
print("gc=%d"%(gc))
print("step=%d"%(step))
print("time=%s"%(time))
print("data width=%d"%(dtype))