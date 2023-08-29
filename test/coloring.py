size = [6, 1, 5]
offset = [1, 1, 1]


color_name = ["black", "red"]

def test(sz):
    detected = False
    black = 0
    red   = 0
    NX = sz[0]
    NY = sz[1]
    NZ = sz[2]
    print("ref color %s"%(color_name[(offset[0] + offset[1] + offset[2]) % 2]))

    print("black indexing")
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                color = (i + j + k + offset[0] + offset[1] + offset[2]) % 2
                idx = i + j * NX + k * NX * NY
                if color == 0:
                    print("%3d -> %3d "%(idx, black), end = "")
                    if black == idx // 2:
                        print(".")
                    else:
                        detected = True
                        print("!")
                    black += 1
    
    print("red indexing")
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                color = (i + j + k + offset[0] + offset[1] + offset[2]) % 2
                idx = i + j * NX + k * NX * NY
                if color == 1:
                    print("%3d -> %3d "%(idx, red), end = "")
                    if red == idx // 2:
                        print(".")
                    else:
                        detected = True
                        print("!")
                    red += 1
    
    print("black=%d red=%d"%(black, red))
    print("indexing error detected: %s"%(detected))

test(size)