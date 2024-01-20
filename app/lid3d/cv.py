cv = open("controlVolume.txt", "w")

L = 1.0
N = 100
d = L / N
gc = 2

cv.write("%d %d %d %d\n"%(N+2*gc, N+2*gc, N+2*gc, gc))

for i in range(-gc, N + gc):
    v = i * d + 0.5 * d
    cv.write("%.15e %.15e\n"%(v, d))

for j in range(-gc, N + gc):
    v = j * d + 0.5 * d
    cv.write("%.15e %.15e\n"%(v, d))

for k in range(-gc, N + gc):
    v = k * d + 0.5 * d
    cv.write("%.15e %.15e\n"%(v, d))