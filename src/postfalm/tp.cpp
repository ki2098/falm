#include <fstream>
#include <sstream>
#include <omp.h>
#include <sys/time.h>
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

void calc_tp(farray<double> &u, farray<double> &p, farray<double> tp, size3 ijkmax, size_t gc) {
    size2 vsz(ijkmax.product(), 3);
    #pragma omp parallel collapse(3) default(shared)
    for (size_t k = gc; k < ijkmax[2] - gc; k ++) {
    for (size_t j = gc; j < ijkmax[1] - gc; j ++) {
    for (size_t i = gc; i < ijkmax[0] - gc; i ++) {
        size_t idx = ijkmax.idx(i, j, k);
        double _u = u[vsz.idx(idx, 0)];
        double _v = u[vsz.idx(idx, 1)];
        double _w = u[vsz.idx(idx, 2)];
        double _p = p[idx];
        tp[idx] = _p + 0.5 * (_u*_u + _v*_v + _w*_w);
    }}}
}

void uvwp_to_tp(string ipath, string opath, farray<double> &u, farray<double> &p, farray<double> &tp, size3 ijkmax, size_t gc) {
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
    fread(ifs, p.ptr(), sizeof(double) * p.size());
    ifs.close();

    calc_tp(u, p, tp, ijkmax, gc);

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
    fwrite(ofs, tp.ptr(), sizeof(double) * tp.size());
    ofs.close();
}