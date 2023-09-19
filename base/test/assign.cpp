#include <stdio.h>
#include "../src/mapper.h"

using namespace Falm;

int main() {
    Mapper a(uint3{5, 5, 6}, uint3{2, 2, 2});
    Mapper b;
    b = a;
    a.offset.x = 1;
    printf("%u %u %u, %u %u %u, %u\n", b.shape.x, b.shape.y, b.shape.z, b.offset.x, b.offset.y, b.offset.z, b.size);

    return 0;
}