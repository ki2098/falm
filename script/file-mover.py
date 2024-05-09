import json
import sys
import os
import shutil

def makefilename(prefix, step):
    return "%s_%010d"%(prefix, step)

srcPrefix = sys.argv[1]
dstPrefix = sys.argv[2]

srcIndexFile = open(srcPrefix+".json")
srcIndexData = json.load(srcIndexFile)
srcIndexFile.close()

for snapshot in srcIndexData["outputSteps"]:
    step = snapshot["step"]
    srcFilename = makefilename(srcPrefix, step)
    dstFilename = makefilename(dstPrefix, step)
    shutil.move(srcFilename, dstFilename)
    print("%s moved to %s"%(srcFilename, dstFilename))

    if "timeAvg" in snapshot:
        srcFilename = makefilename(srcPrefix+"_tavg", step)
        dstFilename = makefilename(dstPrefix+"_tavg", step)
        shutil.move(srcFilename, dstFilename)
        print("%s moved to %s"%(srcFilename, dstFilename))

shutil.move(srcPrefix+".json", dstPrefix+".json")
print("%s moved to %s"%(srcPrefix+".json", dstPrefix+".json"))
shutil.move(srcPrefix+"_setup.json", dstPrefix+"_setup.json")
print("%s moved to %s"%(srcPrefix+"_setup.json", dstPrefix+"_setup.json"))
shutil.move(srcPrefix+".cv", dstPrefix+".cv")
print("%s moved to %s"%(srcPrefix+".cv", dstPrefix+".cv"))
