#include <stdio.h>
#include "../src/matrix.h"
#include "../src/matbasic.h"

using namespace Falm;

int main() {
    Mapper pdom(
        uint3{12, 12, 12},
        uint3{0, 0, 0}
    );
    Mapper map(
        uint3{10, 10, 10},
        uint3{1, 1, 1}
    );
    Matrix<double> a(pdom.shape, 1, HDCTYPE::Host, 0);
    Matrix<double> b(pdom.shape, 1, HDCTYPE::Host, 1);
    for (int i = 0; i < 12; i ++) {
        for (int j = 0; j < 12; j ++) {
            for (int k = 0; k < 12; k ++) {
                unsigned int idx = IDX(i, j, k, pdom.shape);
                a(idx, 0) = idx;
                b(idx, 0) = idx + 1;
            }
        }
    }
    a.sync(MCPTYPE::Hst2Dev);
    b.sync(MCPTYPE::Hst2Dev);
    dim3 block(4, 4, 2);
    double dot  = dev_calc_dot_product(a, b, pdom, map, block);
    double norm = dev_calc_norm2_sq(a, pdom, map, block);
    printf("%.0lf %.0lf\n", dot, norm);

    return 0;
}