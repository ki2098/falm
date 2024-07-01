#include "stdio.h"
#include "stdlib.h"
#include "../src/matrix.h"

using namespace Falm;

Matrix<REAL> *mat;

int main() {
    mat = (Matrix<REAL>*)malloc(sizeof(Matrix<REAL>)*10);
    for (int i = 0; i < 10; i ++) {
        mat[i].alloc(i, i, HDC::HstDev);
    }
    for (int i = 0; i < 10; i ++) {
        printf("(%d %d)\n", mat[i].shape[0], mat[i].shape[1]);
    }
    for (int i = 0; i < 10; i ++) {
        mat[i].release();
    }
    free(mat);
}