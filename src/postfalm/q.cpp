#include <fstream>
#include <sstream>
#include "../nlohmann/json.hpp"
#include "postfalm_array.hpp"
#include "size.hpp"

using json = nlohmann::json;
using namespace std;

void fwrite(ofstream &ofs, void *ptr, size_t sz) {
    ofs.write((char*)ptr, sz);
}

void fread(ifstream &ifs, void *ptr, size_t sz) {
    ifs.read((char*)ptr, sz);
}

void read_cv(string path, farray<double> &x, farray<double> &y, farray<double> &z, size3 &ijkmax, size_t &gc) {
    ifstream ifs(path);
    ifs >> ijkmax[0] >> ijkmax[1] >> ijkmax[2] >> gc;
    x.alloc(ijkmax[0]);
    y.alloc(ijkmax[1]);
    z.alloc(ijkmax[2]);
    double dummy;
    for (size_t i = 0; i < ijkmax[0]; i ++) {
        ifs >> x[i] >> dummy;
    }
    for (size_t j = 0; j < ijkmax[1]; j ++) {
        ifs >> y[j] >> dummy;
    }
    for (size_t k = 0; k < ijkmax[2]; k ++) {
        ifs >> z[k] >> dummy;
    }
}

void calc_q(farray<double> &u, farray<double> &x, farray<double> &y, farray<double> &z, farray<double> &q, size3 ijkmax, size_t gc) {
    size2 vsz(ijkmax.product(), 3);
    for (size_t k = gc; k < ijkmax[2] - gc; k ++) {
    for (size_t j = gc; j < ijkmax[1] - gc; j ++) {
    for (size_t i = gc; i < ijkmax[0] - gc; i ++) {
        size_t idxc = ijkmax.idx(i,j,k);
        size_t idxe = ijkmax.idx(i+1,j,k);
        size_t idxw = ijkmax.idx(i-1,j,k);
        size_t idxn = ijkmax.idx(i,j+1,k);
        size_t idxs = ijkmax.idx(i,j-1,k);
        size_t idxt = ijkmax.idx(i,j,k+1);
        size_t idxb = ijkmax.idx(i,j,k-1);

        double dudx = (u[vsz.idx(idxe, 0)] - u[vsz.idx(idxw, 0)]) / (x[i+1] - x[i-1]);
        double dudy = (u[vsz.idx(idxn, 0)] - u[vsz.idx(idxs, 0)]) / (y[j+1] - y[j-1]);
        double dudz = (u[vsz.idx(idxt, 0)] - u[vsz.idx(idxb, 0)]) / (z[k+1] - z[k-1]);

        double dvdx = (u[vsz.idx(idxe, 1)] - u[vsz.idx(idxw, 1)]) / (x[i+1] - x[i-1]);
        double dvdy = (u[vsz.idx(idxn, 1)] - u[vsz.idx(idxs, 1)]) / (y[j+1] - y[j-1]);
        double dvdz = (u[vsz.idx(idxt, 1)] - u[vsz.idx(idxb, 1)]) / (z[k+1] - z[k-1]);

        double dwdx = (u[vsz.idx(idxe, 2)] - u[vsz.idx(idxw, 2)]) / (x[i+1] - x[i-1]);
        double dwdy = (u[vsz.idx(idxn, 2)] - u[vsz.idx(idxs, 2)]) / (y[j+1] - y[j-1]);
        double dwdz = (u[vsz.idx(idxt, 2)] - u[vsz.idx(idxb, 2)]) / (z[k+1] - z[k-1]);

        q[idxc] = -0.5 * (dudx*dudx + 2*dudy*dvdx + 2*dudz*dwdx + dvdy*dvdy + 2*dvdz*dwdy + dwdz*dwdz);
    }}}
}

void uvw_to_q(string ipath, string opath, farray<double> &u, farray<double> &x, farray<double> &y, farray<double> &z, farray<double> &q, size3 ijkmax, size_t gc) {
    ifstream ifs(ipath, ios::binary);
    size_t imax, jmax, kmax, nv, step, dsz, dgc;
    double time;
    fread(ifs, &imax, sizeof(size_t));
    fread(ifs, &jmax, sizeof(size_t));
    fread(ifs, &kmax, sizeof(size_t));
    fread(ifs, &nv, sizeof(size_t));
    fread(ifs, &dgc, sizeof(size_t));
    fread(ifs, &step, sizeof(size_t));
    fread(ifs, &time, sizeof(double));
    fread(ifs, &dsz, sizeof(size_t));

    assert(imax == ijkmax[0] && jmax == ijkmax[1] && kmax == ijkmax[2] && nv >= 3 && dgc == gc);

    fread(ifs, u.ptr(), sizeof(double) * u.size());
    ifs.close();

    calc_q(u, x, y, z, q, ijkmax, gc);

    ofstream ofs(opath, ios::binary);
    nv = 1;
    fwrite(ofs, &imax, sizeof(size_t));
    fwrite(ofs, &jmax, sizeof(size_t));
    fwrite(ofs, &kmax, sizeof(size_t));
    fwrite(ofs, &nv, sizeof(size_t));
    fwrite(ofs, &dgc, sizeof(size_t));
    fwrite(ofs, &step, sizeof(size_t));
    fwrite(ofs, &time, sizeof(double));
    fwrite(ofs, &dsz, sizeof(size_t));
    fwrite(ofs, q.ptr(), sizeof(double) * q.size());
    ofs.close();
}
