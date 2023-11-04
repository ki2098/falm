#include <stdio.h>
#include "../src/region.h"

using namespace Falm;

int main() {
    Region a(uint3{5, 5, 6}, uint3{2, 2, 2});
    Region b;
    b = a;
    a.offset[0] = 1;
    printf("%u %u %u, %u %u %u, %u\n", b.shape[0], b.shape[1], b.shape[2], b.offset[0], b.offset[1], b.offset[2], b.size);

    return 0;
}
