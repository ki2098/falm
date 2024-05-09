import json
import sys
import os

def makefilename(prefix, step, rank):
    return "%s_%06d_%010d"%(prefix, rank, step)

prefix = sys.argv[1]
indexFile = open(prefix+".json")
data = json.load(indexFile)
indexFile.close()

mpisize = len(data["ranks"])
nsteps = len(data["outputSteps"])

print("%d ranks and %d outputs"%(mpisize, nsteps))

for snapshot in data["outputSteps"]:
    step = snapshot["step"]
    for rank in range(mpisize):
        filename = makefilename(prefix, step, rank)
        if os.path.exists(filename):
            os.remove(filename)
            print("file %s deleted"%(filename))
        else:
            print("ERROR: data file %s does not exist"%(filename))
        if "timeAvg" in snapshot:
            filename = makefilename(prefix+"_tavg", step, rank)
            if os.path.exists(filename):
                os.remove(filename)
                print("file %s deleted"%(filename))
            else:
                print("ERROR: data file %s does not exist"%(filename))

