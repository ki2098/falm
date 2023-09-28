#include <stdio.h>
#include "../src/matrix.h"
#include "../src/MVL1.h"

#define SQR(n) (n)*(n)

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
    Matrix<double> a(pdom.shape, 1, HDCType::Host, 0);
    Matrix<double> b(pdom.shape, 1, HDCType::Host, 1);
    for (int i = 0; i < 12; i ++) {
        for (int j = 0; j < 12; j ++) {
            for (int k = 0; k < 12; k ++) {
                unsigned int idx = IDX(i, j, k, pdom.shape);
                a(idx, 0) = 300 - SQ(i - 7) - SQ(j - 2) - SQ(k - 5);
                b(idx, 0) = 150 - SQ(i - 0) - SQ(j - 1) - SQ(k - 1);
            }
        }
    }
    a.sync(MCpType::Hst2Dev);
    b.sync(MCpType::Hst2Dev);
    dim3 block(4, 4, 2);
    double dot  = L0Dev_DotProduct(a, b, pdom, map, block);
    double norm = L0Dev_Norm2Sq(a, pdom, map, block);
    printf("%.0lf %.0lf\n", dot, norm);

    double a_max = L0Dev_MaxDiag(a, pdom, map, block);
    double b_max = L0Dev_MaxDiag(b, pdom, map, block);
    printf("%.0lf %.0lf\n", a_max, b_max);

    return 0;
}