#include <stdio.h>
#include "../src/matrix.h"
#include "../src/mvbasic.h"

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
                a(idx, 0) = 300 - SQR(i - 7) - SQR(j - 2) - SQR(k - 5);
                b(idx, 0) = 150 - SQR(i - 0) - SQR(j - 1) - SQR(k - 1);
            }
        }
    }
    a.sync(MCpType::Hst2Dev);
    b.sync(MCpType::Hst2Dev);
    dim3 block(4, 4, 2);
    double dot  = dev_DotProduct(a, b, pdom, map, block);
    double norm = dev_Norm2Sq(a, pdom, map, block);
    printf("%.0lf %.0lf\n", dot, norm);

    double a_max = dev_MaxDiag(a, pdom, map, block);
    double b_max = dev_MaxDiag(b, pdom, map, block);
    printf("%.0lf %.0lf\n", a_max, b_max);

    return 0;
}