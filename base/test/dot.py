import numpy as np

def IDX(i, j, k, shape):
    return i + j * shape[0] + k * shape[0] * shape[1]

sumab = 0
sumaa = 0

for i in range(12):
    for j in range(12):
        for k in range(12):
            idx = IDX(i, j, k, (12, 12, 12))
            a = idx
            b = idx + 1
            if (0 < i < 11) and (0 < j < 11) and (0 < k < 11):
                sumab += a * b
                sumaa += a * a

print(sumab, sumaa)