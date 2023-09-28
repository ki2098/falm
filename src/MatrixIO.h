#ifndef FALM_MATRIXIO_H
#define FALM_MATRIXIO_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "matrix.h"

namespace Falm {

template<typename T>
void WriteMatrixToFile(Matrix<T> &matrix, const char *filename) {
    if (!(matrix.hdctype & HDCType::Host)) {
        printf("matrix %s data not present in host memory\n", matrix.cname());
        return;
    }
    FILE *fptr = fopen(filename, "wb");
    if (!fptr) {
        printf("can't open file %s\n", filename);
        return;
    }
    if(fwrite(&matrix.shape, sizeof(INTx2), 1, fptr) < 1) {
        printf("error writing matrix %s shape\n", matrix.cname());
        fclose(fptr);
        return;
    }
    if (fwrite(matrix.host.ptr, sizeof(T), matrix.size, fptr) < matrix.size) {
        printf("error writing matrix %s data\n", matrix.cname());
        fclose(fptr);
        return;
    } 
    fclose(fptr);
    return;
}

template<typename T>
void ReadMatrixFromFile(Matrix<T> &matrix, const char *filename) {
    if (matrix.hdctype != HDCType::Empty) {
        printf("matrix %s is occupied, can't load data from %s\n", matrix.cname(), filename);
        return;
    }
    FILE *fptr = fopen(filename, "rb");
    FILE *fptr = fopen(filename, "wb");
    if (!fptr) {
        printf("can't open file %s\n", filename);
        return;
    }
    INTx2 shape;
    if (fread(&shape, sizeof(INTx2), 1, fptr) < 1) {
        printf("error reading matrix %s shape\n", matrix.cname());
        fclose(fptr);
        return;
    }
    matrix.alloc(shape.x, shape.y, HDCType::Host);
    if (fread(matrix.host.ptr, sizeof(T), matrix.size, fptr) < matrix.size) {
        printf("error reading matrix %s data\n", matrix.cname());
        fclose(fptr);
        return;
    }
    fclose(fptr);
    return;
}

}

#endif