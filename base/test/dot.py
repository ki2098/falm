import numpy as np

def SQR(n):
    return n * n

def IDX(i, j, k, shape):
    return i + j * shape[0] + k * shape[0] * shape[1]

sumab = 0
sumaa = 0

maxa = 0
maxb = 0

for i in range(12):
    for j in range(12):
        for k in range(12):
            idx = IDX(i, j, k, (12, 12, 12))
            a = 300 - SQR(i - 7) - SQR(j - 2) - SQR(k - 5)
            b = 150 - SQR(i - 0) - SQR(j - 1) - SQR(k - 1)
            if (0 < i < 11) and (0 < j < 11) and (0 < k < 11):
                sumab += a * b
                sumaa += a * a
                if (np.abs(a) > maxa):
                    maxa = np.abs(a)
                if (np.abs(b) > maxb):
                    maxb = np.abs(b)

print(sumab, sumaa)
print(maxa, maxb)