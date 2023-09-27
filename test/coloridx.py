world_shape  = [7, 7, 7]
world_offset = [0, 0, 0]
buf_shape    = [1, 5, 5]
buf_offset   = [6, 2, 2]

def IDX(i, j, k, shape):
    return i + j * shape[0] + k * shape[0] * shape[1]

def SUM3(arr):
    return arr[0] + arr[1] + arr[2]

buf_red   = 0
buf_black = 0
for k in range(buf_shape[2]):
    for j in range(buf_shape[1]):
        for i in range(buf_shape[0]):
            buf_idx = IDX(i, j, k, buf_shape)
            wi = i + buf_offset[0]
            wj = j + buf_offset[1]
            wk = k + buf_offset[2]
            world_idx = IDX(wi, wj, wk, world_shape)
            color = (wi + wj + wk + SUM3(world_offset)) % 2
            if color == 0:
                true_idx = buf_black
                buf_black += 1
            else:
                true_idx = buf_red
                buf_red += 1
            print("%5d %5d %2d %5d"%(buf_idx, world_idx, color, true_idx))
            
