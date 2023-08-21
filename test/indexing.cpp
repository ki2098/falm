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
    unsigned int g = 2;
    unsigned int isz[3] = {3, 3, 3};
    unsigned int osz[3] = {isz[0]+2*g, isz[1]+2*g, isz[2]+2*g};
    for (int idx = 0; idx < 27; idx ++) {
        unsigned int ii, ij, ik;
        idx2ijk(ii, ij, ik, idx, isz);
        unsigned int oi, oj, ok;
        oi = ii + g;
        oj = ij + g;
        ok = ik + g;
        unsigned int odx = ijk2idx(oi, oj, ok, osz);
        printf("inner %4d outer %4d\n", idx, odx);
    }
    return 0;
}