#include <cstdio>
#include <cstdlib>

unsigned int ijk2idx(unsigned int i, unsigned int j, unsigned int k, unsigned int size[3]) {
    return i * size[1] * size[2] + j * size[2] + k;
}

void idx2ijk(unsigned int &i, unsigned int &j, unsigned int &k, unsigned int idx, unsigned int size[3]) {
    unsigned int slice = size[1] * size[2];
    i   = idx / slice;
    idx = idx % slice;
    j   = idx / size[2];
    k   = idx % size[2];
}

int main() {
    unsigned int size[3] = {123, 252, 332};
    unsigned int i = 55;
    unsigned int j = 71;
    unsigned int k = 101;
    printf("3d ijk = %u %u %u\n", i, j, k);
    unsigned int idx = ijk2idx(i, j, k, size);
    printf("1d idx = %u\n", idx);
    idx2ijk(i, j, k, idx, size);
    printf("3d ijk = %u %u %u\n", i, j, k);
}