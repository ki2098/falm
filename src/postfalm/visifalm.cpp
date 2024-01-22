#include <fstream>
#include <sstream>
#include "../nlohmann/json.hpp"
#include "postfalm_array.hpp"
#include "size.hpp"

using json = nlohmann::json;
using namespace std;


struct SphDummyGrid {
    double xorg, yorg, zorg, xpitch, ypitch, zpitch;
};

SphDummyGrid sdg;

struct TimeSliceInfo {
    size_t step;
    double time;
    bool tavg;
};

vector<TimeSliceInfo> slice_list;

void read_index_file(string path) {
    ifstream idxfile(path);
    json idxjson = json::parse(idxfile);
    for (auto slice : idxjson["outputSteps"]) {
        TimeSliceInfo tsinfo;
        tsinfo.step = slice["step"].get<size_t>();
        tsinfo.time = slice["time"].get<double>();
        tsinfo.tavg = false;
        if (slice.contains("timeAvg")) {
            tsinfo.tavg = true;
        }
        // pair<size_t, double> pt{_step, _time};
        slice_list.push_back(tsinfo);
    }
}

void record(ofstream &ofs, int sz) {
    ofs.write((char*)&sz, sizeof(int));
}

void fwrite(ofstream &ofs, void *ptr, size_t sz) {
    ofs.write((char*)ptr, sz);
}

void fread(ifstream &ifs, void *ptr, size_t sz) {
    ifs.read((char*)ptr, sz);
}

string make_filename(string prefix, size_t step) {
    farray<char> tmpc(prefix.size()+32);
    sprintf(tmpc.ptr(), "%s_%010lu", prefix.c_str(), step);
    return string(tmpc.ptr());
}

void write_sph(double *data, string path, size4 size, size_t gc, size_t step, double time) {
    ofstream ofs(path, ios::binary);
    int svtype, dtype, rsz;
    if (size[3] == 1) {
        svtype = 1;
    } else if (size[3] == 3) { 
        svtype = 2;
    } else {
        svtype = 2;
        printf("wrong number of variables %ld in SPH file %s\n", size[3], path.c_str());
    }
    dtype = 1;

    record(ofs, 2*sizeof(int));
    fwrite(ofs, &svtype, sizeof(int));
    fwrite(ofs, &dtype, sizeof(int));
    record(ofs, 2*sizeof(int));

    int _imax = size[0]-2*gc, _jmax = size[1]-2*gc, _kmax = size[2]-2*gc;
    record(ofs, 3*sizeof(int));
    fwrite(ofs, &_imax, sizeof(int));
    fwrite(ofs, &_jmax, sizeof(int));
    fwrite(ofs, &_kmax, sizeof(int));
    record(ofs, 3*sizeof(int));

    float xorg = sdg.xorg, yorg = sdg.yorg, zorg = sdg.zorg;
    float xpitch = sdg.xpitch, ypitch = sdg.ypitch, zpitch = sdg.zpitch;
    record(ofs, 3*sizeof(float));
    fwrite(ofs, &xorg, sizeof(float));
    fwrite(ofs, &yorg, sizeof(float));
    fwrite(ofs, &zorg, sizeof(float));
    record(ofs, 3*sizeof(float));

    record(ofs, 3*sizeof(float));
    fwrite(ofs, &xpitch, sizeof(float));
    fwrite(ofs, &ypitch, sizeof(float));
    fwrite(ofs, &zpitch, sizeof(float));
    record(ofs, 3*sizeof(float));

    int _step = step;
    float _time = time;
    record(ofs, sizeof(int)+sizeof(float));
    fwrite(ofs, &_step, sizeof(int));
    fwrite(ofs, &_time, sizeof(float));
    record(ofs, sizeof(int)+sizeof(float));

    record(ofs, sizeof(float)*_imax*_jmax*_kmax*size[3]);
    for (size_t k = gc; k < size[2]-gc; k ++) {
    for (size_t j = gc; j < size[1]-gc; j ++) {
    for (size_t i = gc; i < size[0]-gc; i ++) {
    for (size_t n = 0; n < size[3]; n ++) {
        float v = data[size.idx(i, j, k, n)];
        fwrite(ofs, &v, sizeof(float));
    }}}}
    record(ofs, sizeof(float)*_imax*_jmax*_kmax*size[3]);

    ofs.close();
}

void uvwp_to_sph(string prefix, size_t step, double time, bool cut_gc) {
    string uvwprefix = prefix + "_velocity";
    string pprefix = prefix + "_pressure";
    string ifname = make_filename(prefix, step);
    printf("converting %s to %s and %s...\n", ifname.c_str(), uvwprefix.c_str(), pprefix.c_str());
    fflush(stdout);

    size_t imax, jmax, kmax, nv, gc, _step, dtype;
    double _time;

    ifstream ifs(ifname);
    fread(ifs, &imax, sizeof(size_t));
    fread(ifs, &jmax, sizeof(size_t));
    fread(ifs, &kmax, sizeof(size_t));
    fread(ifs, &nv, sizeof(size_t));
    fread(ifs, &gc, sizeof(size_t));
    fread(ifs, &_step, sizeof(size_t));
    fread(ifs, &_time, sizeof(double));
    fread(ifs, &dtype, sizeof(size_t));
    if (dtype != sizeof(double)) {
        printf("wrong data width %lu in %s\n", dtype, prefix.c_str());
    }

    size4 size(imax, jmax, kmax, nv);
    farray<double> v(size.product());

    fread(ifs, v.ptr(), sizeof(double)*v.size());

    string uvwfname = make_filename(uvwprefix, step) + ".sph";
    string pfname = make_filename(pprefix, step) + ".sph";

    write_sph(&v[size.idx(0, 0, 0, 0)], uvwfname, size4(imax, jmax, kmax, 3), gc*cut_gc, step, time);
    write_sph(&v[size.idx(0, 0, 0, 3)], pfname, size4(imax, jmax, kmax, 1), gc*cut_gc, step, time);

    ifs.close();
}

void cv_to_crd(string ipath, string opath, bool cut_gc) {
    printf("converting %s to %s\n", ipath.c_str(), opath.c_str());fflush(stdout);
    fflush(stdout);
    ifstream ifs(ipath);
    size_t imax, jmax, kmax, gc;
    ifs >> imax >> jmax >> kmax >> gc;
    int gc_mask = int(cut_gc);

    farray<double> x(imax);
    farray<double> y(jmax);
    farray<double> z(kmax);
    farray<double> hx(imax);
    farray<double> hy(jmax);
    farray<double> hz(kmax);
    for (size_t i = 0; i < imax; i ++) {
        ifs >> x[i] >> hx[i];
    }
    for (size_t j = 0; j < jmax; j ++) {
        ifs >> y[j] >> hy[j];
    }
    for (size_t k = 0; k < kmax; k ++) {
        ifs >> z[k] >> hz[k];
    }
    ifs.close();

    ofstream ofs(opath, ios::binary);
    int dsz = 1;

    record(ofs, sizeof(int)*2);
    fwrite(ofs, &dsz, sizeof(int));
    fwrite(ofs, &dsz, sizeof(int));
    record(ofs, sizeof(int)*2);

    int xnum = imax - 2*gc*gc_mask + 1;
    int ynum = jmax - 2*gc*gc_mask + 1;
    int znum = kmax - 2*gc*gc_mask + 1;
    record(ofs, sizeof(int)*3);
    fwrite(ofs, &xnum, sizeof(int));
    fwrite(ofs, &ynum, sizeof(int));
    fwrite(ofs, &znum, sizeof(int));
    record(ofs, sizeof(int)*3);

    int   step = 0;
    float time = 0;
    record(ofs, sizeof(int)+sizeof(float));
    fwrite(ofs, &step, sizeof(int));
    fwrite(ofs, &time, sizeof(float));
    record(ofs, sizeof(int)+sizeof(float));

    size_t offset = gc*gc_mask;

    record(ofs, sizeof(float)*xnum);
    for (size_t i = 0; i < xnum; i ++) {
        size_t tmp = i + offset;
        float v;
        if (tmp < imax) {
            v = x[tmp] - 0.5 * hx[tmp];
        } else {
            v = x[tmp-1] + 0.5 * hx[tmp-1];
        }
        fwrite(ofs, &v, sizeof(float));
        if (i == 0) {
            sdg.xorg = v;
        } else if (i == xnum - 1) {
            sdg.xpitch = (v - sdg.xorg) / (xnum - 1);
        }
    }
    record(ofs, sizeof(float)*xnum);

    record(ofs, sizeof(float)*ynum);
    for (size_t j = 0; j < ynum; j ++) {
        size_t tmp = j + offset;
        float v;
        if (tmp < jmax) {
            v = y[tmp] - 0.5 * hy[tmp];
        } else {
            v = y[tmp-1] + 0.5 * hy[tmp-1];
        }
        fwrite(ofs, &v, sizeof(float));
        if (j == 0) {
            sdg.yorg = v;
        } else if (j == ynum - 1) {
            sdg.ypitch = (v - sdg.yorg) / (ynum - 1);
        }
    }
    record(ofs, sizeof(float)*ynum);

    record(ofs, sizeof(float)*znum);
    for (size_t k = 0; k < znum; k ++) {
        size_t tmp = k + offset;
        float v;
        if (tmp < kmax) {
            v = z[tmp] - 0.5 * hz[tmp];
        } else {
            v = z[tmp-1] + 0.5 * hz[tmp-1];
        }
        fwrite(ofs, &v, sizeof(float));
        if (k == 0) {
            sdg.zorg = v;
        } else if (k == znum - 1) {
            sdg.zpitch = (v - sdg.zorg) / (znum - 1);
        }
    }
    record(ofs, sizeof(float)*znum);

    ofs.close();
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("too few arguments.\n");
        return 0;
    }
    string prefix(argv[1]);
    if (argc > 2) {
        vector<size_t> convert_list;
        for (int i = 2; i < argc; i ++) {

        }
    }
    printf("VisiFalm: converting %s files to Riken V-Isio files\n", prefix.c_str());
    read_index_file(prefix+".json");
    cv_to_crd(prefix + ".cv", prefix + ".crd", true);
    for (auto slice : slice_list) {
        size_t step = slice.step;
        double time = slice.time;

        uvwp_to_sph(prefix, step, time, true);

        if (slice.tavg) {
            uvwp_to_sph(prefix + "_tavg", step, time, true);
        }
    }

    return 0;
}