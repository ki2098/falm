import numpy as np

def getvalues(str: str):
    values = [float(item) for item in str.split() if item]
    return values

class AP:
    def __init__(self):
        self.r = 0.
        self.chord = 0.
        self.twist = 0.
        self.attack = None
        self.cl = None
        self.cd = None


def fill_chord_twist(aps: list[AP], rr, chord, twist):
    for ap in aps:
        if ap.r < rr[0]:
            ap.chord = chord[0]
            ap.twist = twist[0]
        elif ap.r >= rr[-1]:
            ap.chord = chord[-1]
            ap.twist = twist[-1]
        else:
            for i in range(len(rr)):
                if rr[i] <= ap.r < rr[i+1]:
                    pp = (ap.r - rr[i])/(rr[i+1] - rr[i])
                    ap.chord = (1 - pp)*chord[i] + pp*chord[i+1]
                    ap.twist = (1 - pp)*twist[i] + pp*twist[i+1]

def fill_cl_cd(aps: list[AP], rr, attack, cl, cd):
    for ap in aps:
        ap.attack = attack
        if ap.r < rr[0]:
            ap.cl = cl[:,0]
            ap.cd = cd[:,0]
        elif ap.r >= rr[-1]:
            ap.cl = cl[:,-1]
            ap.cd = cd[:,-1]
        else:
            for i in range(len(rr)):
                if rr[i] <= ap.r < rr[i+1]:
                    pp = (ap.r - rr[i])/(rr[i+1] - rr[i])
                    ap.cl = (1 - pp)*cl[:,i] + pp*cl[:,i+1]
                    ap.cd = (1 - pp)*cd[:,i] + pp*cd[:,i+1]

NAPPB = 20
NBPT = 3
NAP = 120
NAPPT = NAPPB*NBPT
aps = [AP() for i in range(NAP)]
for i in range(NAP):
    api = i%NAPPB+1
    aps[i].r = api*(1.0/NAPPB)

f = open("bladeParameter", "r")

rr = []
chord = []
twist = []
f.readline()
while True:
    values = getvalues(f.readline())
    if len(values) == 0:
        break
    rr.append(values[0])
    chord.append(values[1])
    twist.append(values[2])

fill_chord_twist(aps, rr, chord, twist)

f.readline()
rr = getvalues(f.readline())
nrr = len(rr)
attack = []
cl = []
cd = []
f.readline()
while True:
    values = getvalues(f.readline())
    if len(values) == 0:
        break
    attack.append(values[0])
    cl.append([values[i*2+1] for i in range(nrr)])
    cd.append([values[i*2+2] for i in range(nrr)])

cl = np.array(cl)
cd = np.array(cd)

fill_cl_cd(aps, rr, attack, cl, cd)

f.close()

f = open("apparameter", "w")
f.write("# AP count\n%d\n"%(len(aps)))
f.write("# attack angle count\n%d\n\n"%(len(attack)))
for i in range(NAP):
    ap = aps[i]
    f.write("# AP id\n%d\n"%(i))
    f.write("# turbine id\n%d\n"%(i//NAPPT))
    f.write("# blade id\n%d\n"%((i%NAPPT)//NAPPB))
    f.write("# r/R\n%13e\n"%(ap.r))
    f.write("# chord/R\n%13e\n"%(ap.chord))
    f.write("# twist\n%13e\n"%(ap.twist))
    f.write("# attack cl cd\n")
    for a in range(len(ap.attack)):
        f.write("%5s %13e %13e\n"%(ap.attack[a], ap.cl[a], ap.cd[a]))
    f.write("\n")
f.close()
