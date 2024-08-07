#include <stdio.h>
#include <string>
#include <fstream>
#include <cstring>

using namespace std;

size_t imax, jmax, kmax, nmax, gc, step, type;
double tt;

size_t id(size_t i, size_t j, size_t k, size_t n) {
    return n*imax*jmax*kmax + k*imax*jmax + j*imax + i;
}

size_t id(size_t i, size_t j, size_t k) {
    return k*imax*jmax + j*imax + i;
}

int main() {
    string prefix = "data/uvwp2";
    string fname = prefix + "_0000015000";
    string cvname = prefix + ".cv";
    FILE *file = fopen(fname.c_str(), "rb");

    size_t dumm;
    
    dumm = fread(&imax, sizeof(size_t), 1, file);
    dumm = fread(&jmax, sizeof(size_t), 1, file);
    dumm = fread(&kmax, sizeof(size_t), 1, file);
    dumm = fread(&nmax, sizeof(size_t), 1, file);
    dumm = fread(&gc  , sizeof(size_t), 1, file);
    dumm = fread(&step, sizeof(size_t), 1, file);
    dumm = fread(&tt  , sizeof(double), 1, file);
    dumm = fread(&type, sizeof(size_t), 1, file);

    double *data = (double*)malloc(sizeof(double)*imax*jmax*kmax*nmax);
    dumm = fread(data, sizeof(double), imax*jmax*kmax*nmax, file);

    fclose(file);

    ifstream ifs(cvname);
    double *x = (double*)malloc(sizeof(double)*imax);
    double *y = (double*)malloc(sizeof(double)*jmax);
    double *z = (double*)malloc(sizeof(double)*kmax);
    double dummy;
    ifs >> dummy >> dummy >> dummy >> dummy;
    for (size_t i = 0; i < imax; i ++) {
        ifs >> x[i] >> dummy;
    }
    for (size_t j = 0; j < jmax; j ++) {
        ifs >> y[j] >> dummy;
    }
    for (size_t k = 0; k < kmax; k ++) {
        ifs >> z[k] >> dummy;
    }
    ifs.close();

    double *q = (double*)malloc(sizeof(double)*imax*jmax*kmax);
    memset(q, 0, sizeof(double)*imax*jmax*kmax);

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < imax-1; i ++) {
    for (int j = 1; j < jmax-1; j ++) {
    for (int k = 1; k < kmax-1; k ++) {
        double ue = data[id(i+1,j,k,0)];
        double uw = data[id(i-1,j,k,0)];
        double un = data[id(i,j+1,k,0)];
        double us = data[id(i,j-1,k,0)];
        double ut = data[id(i,j,k+1,0)];
        double ub = data[id(i,j,k-1,0)];

        double ve = data[id(i+1,j,k,1)];
        double vw = data[id(i-1,j,k,1)];
        double vn = data[id(i,j+1,k,1)];
        double vs = data[id(i,j-1,k,1)];
        double vt = data[id(i,j,k+1,1)];
        double vb = data[id(i,j,k-1,1)];

        double we = data[id(i+1,j,k,2)];
        double ww = data[id(i-1,j,k,2)];
        double wn = data[id(i,j+1,k,2)];
        double ws = data[id(i,j-1,k,2)];
        double wt = data[id(i,j,k+1,2)];
        double wb = data[id(i,j,k-1,2)];

        double xe = x[i+1];
        double xw = x[i-1];
        double yn = y[j+1];
        double ys = y[j-1];
        double zt = z[k+1];
        double zb = z[k-1];

        double dudx = (ue - uw)/(xe - xw);
        double dudy = (un - us)/(yn - ys);
        double dudz = (ut - ub)/(zt - zb);

        double dvdx = (ve - vw)/(xe - xw);
        double dvdy = (vn - vs)/(yn - ys);
        double dvdz = (vt - vb)/(zt - zb);

        double dwdx = (we - ww)/(xe - xw);
        double dwdy = (wn - ws)/(yn - ys);
        double dwdz = (wt - wb)/(zt - zb);

        q[id(i,j,k)] = - 0.5*(dudx*dudx + dvdy*dvdy + dwdz*dwdz + 2*(dudy*dvdx + dudz*dwdx + dvdz*dwdy));
    }}}

    file = fopen((fname+".csv").c_str(), "w");

    fprintf(file, "x,y,z,u,v,w,p,q\n");
    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        fprintf(file, "%e,%e,%e,%e,%e,%e,%e,%e\n", x[i], y[j], z[k], data[id(i,j,k,0)], data[id(i,j,k,1)], data[id(i,j,k,2)], data[id(i,j,k,3)], q[id(i,j,k)]);
    }}}
    printf("%lu %lu %lu\n", imax, jmax, kmax);

    free(data);
    free(x);
    free(y);
    free(z);
    free(q);
    fclose(file);
}