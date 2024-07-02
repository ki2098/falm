#include <stdio.h>
#include <string>

using namespace std;

size_t imax, jmax, kmax, nmax, gc, step, type;
double tt;

size_t id(size_t i, size_t j, size_t k, size_t n) {
    return n*imax*jmax*kmax + k*imax*jmax + j*imax + i;
}

int main() {
    string fname = "data/uvwp_000000_0000030000";
    FILE *file = fopen(fname.c_str(), "rb");
    
    fread(&imax, sizeof(size_t), 1, file);
    fread(&jmax, sizeof(size_t), 1, file);
    fread(&kmax, sizeof(size_t), 1, file);
    fread(&nmax, sizeof(size_t), 1, file);
    fread(&gc  , sizeof(size_t), 1, file);
    fread(&step, sizeof(size_t), 1, file);
    fread(&tt  , sizeof(double), 1, file);
    fread(&type, sizeof(size_t), 1, file);

    double *data = (double*)malloc(sizeof(double)*imax*jmax*kmax*nmax);
    fread(data, sizeof(double), imax*jmax*kmax*nmax, file);

    fclose(file);

    file = fopen((fname+".csv").c_str(), "w");

    fprintf(file, "i,j,k,u,v,w,p\n");
    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        fprintf(file, "%lu,%lu,%lu,%e,%e,%e,%e\n", i, j, k, data[id(i,j,k,0)], data[id(i,j,k,1)], data[id(i,j,k,2)], data[id(i,j,k,3)]);
    }}}
    printf("%lu %lu %lu\n", imax, jmax, kmax);

    free(data);
    fclose(file);
}