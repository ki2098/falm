#include <stdio.h>
#include "../base/src/matrix.h"


typedef uint2 uintx2;
typedef uint3 uintx3;

struct X {
    uintx3 u3;
    uintx2 u2;
};

int main() {
    printf("%lu\n", sizeof(Falm::MatrixFrame<double>));

    X xs[5];
    for (int i = 0; i < 5; i ++) {
        printf("%p\n", &xs[i]);
    }

    return 0;
}