#include <fstream>
#include <sstream>
#include <omp.h>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;

struct size3 {
    size_t _sz[3];
    size3(size_t _vx, size_t _vy, size_t _vz) {
        _sz[0] = _vx;
        _sz[1] = _vy;
        _sz[2] = _vz;
    }

    size3(const json &jsz) {
        _sz[0] = jsz[0];
        _sz[1] = jsz[1];
        _sz[2] = jsz[2];
    }

    size3() {}

    size_t &operator[](size_t i) {
        return _sz[i];
    }

    const size_t &operator[](size_t i) const {
        return _sz[i];
    }

    size3 operator+(size_t _v) {
        size3 tmp;
        tmp[0] = _sz[0] + _v;
        tmp[1] = _sz[1] + _v;
        tmp[2] = _sz[2] + _v;
        return tmp;
    }

    std::string str() {
        stringstream tmp;
        tmp << "(" << _sz[0] << " " << _sz[1] << " " << _sz[2] << ")";
        return tmp.str();
    }

    size_t product() const {
        return _sz[0] * _sz[1] * _sz[2];
    }

    size_t idx(size_t i, size_t j, size_t k) {
        return i + j * _sz[0] + k * _sz[0] * _sz[1];
    }
};

struct size4 {
    size_t _sz[4];

    size4(size_t _vx, size_t _vy, size_t _vz, size_t _vn) {
        _sz[0] = _vx;
        _sz[1] = _vy;
        _sz[2] = _vz;
        _sz[3] = _vn;
    }

    size4(const size3 &sz3, size_t n) {
        _sz[0] = sz3[0];
        _sz[1] = sz3[1];
        _sz[2] = sz3[2];
        _sz[3] = n;
    }

    size_t &operator[](size_t i) {
        return _sz[i];
    }

    const size_t &operator[](size_t i) const {
        return _sz[i];
    }

    size3 operator+(size_t _v) {
        size3 tmp;
        tmp[0] = _sz[0] + _v;
        tmp[1] = _sz[1] + _v;
        tmp[2] = _sz[2] + _v;
        return tmp;
    }

    size_t product() const {
        return _sz[0] * _sz[1] * _sz[2] * _sz[3];
    }

    size_t idx(size_t i, size_t j, size_t k, size_t n) {
        return i + j * _sz[0] + k * _sz[0] * _sz[1] + n * _sz[0] * _sz[1] * _sz[2];
    }
};

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

size_t idx_ijkn(size_t imax, size_t jmax, size_t kmax, size_t nmax, size_t i, size_t j, size_t k, size_t n) {
    return i + j*imax + k*imax*jmax + n*imax*jmax*kmax;
}

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
    char *tmp = (char*)malloc(sizeof(char) * (prefix.size() + 32));
    sprintf(tmp, "%s_%010d", prefix.c_str(), step);
    string str(tmp);
    free(tmp);
    return str;
}

void write_sph_float(double *data, string path, size_t imax, size_t jmax, size_t kmax, size_t nv, size_t gc, size_t step, double time) {
    ofstream ofs(path, ios::binary);
    int svtype, dtype, rsz;

    size4 size(imax, jmax, kmax, nv);

    if (nv == 1) svtype = 1;
    else if (nv == 3) svtype = 2;
    else {
        svtype = 2;
        printf("wrong number of variables %ld in SPH file %s\n", nv, path.c_str());
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

    record(ofs, sizeof(float)*_imax*_jmax*_kmax*nv);
    #pragma omp parallel collapse(4)
    for (size_t k = gc; k < size[2]-gc; k ++) {
    for (size_t j = gc; j < size[1]-gc; j ++) {
    for (size_t i = gc; i < size[0]-gc; i ++) {
    for (size_t n = 0; n < size[3]; n ++) {
        float v = data[size.idx(i, j, k, n)];
        fwrite(ofs, &v, sizeof(float));
    }}}}
    record(ofs, sizeof(float)*_imax*_jmax*_kmax*nv);

    ofs.close();
}

template<typename T>
void uvwp_to_sph(string prefix, bool cut_gc) {
    for (auto slice : slice_list) {
        
    }
}

template<typename T>
void cvnode_to_crd(string ipath, string opath, bool cut_gc) {
    printf("converting %s to %s\n", ipath.c_str(), opath.c_str());fflush(stdout);
    ifstream ifs(ipath);
    size_t imax, jmax, kmax, gc;
    ifs >> imax >> jmax >> kmax >> gc;
    int gc_mask = int(cut_gc);
    double *x, *y, *z, *hx, *hy, *hz;
    x = (double*)malloc(sizeof(double)*imax);
    y = (double*)malloc(sizeof(double)*jmax);
    z = (double*)malloc(sizeof(double)*kmax);
    hx = (double*)malloc(sizeof(double)*imax);
    hy = (double*)malloc(sizeof(double)*jmax);
    hz = (double*)malloc(sizeof(double)*kmax);
    for (size_t i = 0; i < imax; i ++) {
        ifs >> x[i] >> hx[i];
    }
    for (size_t j = 0; j < jmax; j ++) {
        ifs >> y[j] >> hy[j];
    }
    for (size_t k = 0; k < kmax; k ++) {
        ifs >> z[k] >> hz[k];
    }

    if (typeid(T) == typeid(float)) {
        ofstream ofs(opath, ios::binary);
        int rsz, dsz = 1;
        
        rsz = 2 * sizeof(int);
        record(ofs, rsz);
        ofs.write((char*)&dsz, sizeof(int));
        ofs.write((char*)&dsz, sizeof(int));
        record(ofs, rsz);

        rsz = 3 * sizeof(int);
        int xnum = imax - 2*gc*gc_mask + 1;
        int ynum = jmax - 2*gc*gc_mask + 1;
        int znum = kmax - 2*gc*gc_mask + 1;
        record(ofs, rsz);
        ofs.write((char*)&xnum, sizeof(int));
        ofs.write((char*)&ynum, sizeof(int));
        ofs.write((char*)&znum, sizeof(int));
        record(ofs, rsz);

        rsz = sizeof(int) + sizeof(float);
        int   step = 0;
        float time = 0;
        record(ofs, rsz);
        ofs.write((char*)&step, sizeof(int));
        ofs.write((char*)&time, sizeof(float));
        record(ofs, rsz);

        size_t offset = gc*gc_mask;

        rsz = sizeof(float) * xnum;
        record(ofs, rsz);
        for (size_t i = 0; i < xnum; i ++) {
            size_t tmp = i + offset;
            float v;
            if (tmp < imax) {
                v = x[tmp] - 0.5 * hx[tmp];
            } else {
                v = x[tmp-1] + 0.5 * hx[tmp-1];
            }
            ofs.write((char*)&v, sizeof(float));
            if (i == 0) {
                sdg.xorg = v;
            } else if (i == xnum - 1) {
                sdg.xpitch = (v - sdg.xorg) / (xnum - 1);
            }
        }
        record(ofs, rsz);

        rsz = sizeof(float) * ynum;
        record(ofs, rsz);
        for (size_t j = 0; j < ynum; j ++) {
            size_t tmp = j + offset;
            float v;
            if (tmp < jmax) {
                v = y[tmp] - 0.5 * hy[tmp];
            } else {
                v = y[tmp-1] + 0.5 * hy[tmp-1];
            }
            ofs.write((char*)&v, sizeof(float));
            if (j == 0) {
                sgd.yorg = v;
            } else if (j == ynum - 1) {
                sgd.ypitch = (v - sgd.yorg) / (ynum - 1);
            }
        }
        record(ofs, rsz);

        rsz = sizeof(float) * znum;
        record(ofs, rsz);
        for (size_t k = 0; k < znum; k ++) {
            size_t tmp = k + offset;
            float v;
            if (tmp < kmax) {
                v = z[tmp] - 0.5 * hz[tmp];
            } else {
                v = z[tmp-1] + 0.5 * hz[tmp-1];
            }
            ofs.write((char*)&v, sizeof(float));
            if (k == 0) {
                sdg.zorg = v;
            } else if (k == znum - 1) {
                sdg.zpitch = (v - sdg.zorg) / (znum - 1);
            }
        }
        record(ofs, rsz);
        ofs.close();
    } else if (typeid(T) == typeid(double)) {
        ofstream ofs(opath, ios::binary);
        int rsz, dsz = 1;
        
        rsz = 2 * sizeof(int);
        record(ofs, rsz);
        ofs.write((char*)&dsz, sizeof(int));
        ofs.write((char*)&dsz, sizeof(int));
        record(ofs, rsz);

        rsz = 3 * sizeof(int);
        int64_t xnum = imax - 2*gc*gc_mask + 1;
        int64_t ynum = jmax - 2*gc*gc_mask + 1;
        int64_t znum = kmax - 2*gc*gc_mask + 1;
        record(ofs, rsz);
        ofs.write((char*)&xnum, sizeof(int64_t));
        ofs.write((char*)&ynum, sizeof(int64_t));
        ofs.write((char*)&znum, sizeof(int64_t));
        record(ofs, rsz);

        rsz = sizeof(int64_t) + sizeof(double);
        int64_t step = 0;
        double  time = 0;
        record(ofs, rsz);
        ofs.write((char*)&step, sizeof(int64_t));
        ofs.write((char*)&time, sizeof(double));
        record(ofs, rsz);

        size_t offset = gc*gc_mask;

        rsz = sizeof(double) * xnum;
        record(ofs, rsz);
        for (size_t i = 0; i < xnum; i ++) {
            size_t tmp = i + offset;
            double v;
            if (tmp < imax) {
                v = x[tmp] - 0.5 * hx[tmp];
            } else {
                v = x[tmp-1] + 0.5 * hx[tmp-1];
            }
            ofs.write((char*)&v, sizeof(double));
        }
        record(ofs, rsz);

        rsz = sizeof(double) * ynum;
        record(ofs, rsz);
        for (size_t j = 0; j < ynum; j ++) {
            size_t tmp = j + offset;
            double v;
            if (tmp < jmax) {
                v = y[tmp] - 0.5 * hy[tmp];
            } else {
                v = y[tmp-1] + 0.5 * hy[tmp-1];
            }
            ofs.write((char*)&v, sizeof(double));
        }
        record(ofs, rsz);

        rsz = sizeof(double) * znum;
        record(ofs, rsz);
        for (size_t k = 0; k < znum; k ++) {
            size_t tmp = k + offset;
            double v;
            if (tmp < kmax) {
                v = z[tmp] - 0.5 * hz[tmp];
            } else {
                v = z[tmp-1] + 0.5 * hz[tmp-1];
            }
            ofs.write((char*)&v, sizeof(double));
        }
        record(ofs, rsz);
        ofs.close();
    }

    ifs.close();
    free(x);
    free(y);
    free(z);
    free(hx);
    free(hy);
    free(hz);
}

template<typename T>
void cvcenter_to_crd(string ipath, string opath, bool cut_gc) {
    printf("converting %s to %s\n", ipath.c_str(), opath.c_str());fflush(stdout);
    ifstream ifs(ipath);
    size_t imax, jmax, kmax, gc;
    ifs >> imax >> jmax >> kmax >> gc;
    int gc_mask = int(cut_gc);
    if (typeid(T) == typeid(float)) {
        float *x, *y, *z, dummy;
        x = (float*)malloc(sizeof(float)*imax);
        y = (float*)malloc(sizeof(float)*jmax);
        z = (float*)malloc(sizeof(float)*kmax);
        for (size_t i = 0; i < imax; i ++) {
            ifs >> x[i] >> dummy;
        }
        for (size_t j = 0; j < jmax; j ++) {
            ifs >> y[j] >> dummy;
        }
        for (size_t k = 0; k < kmax; k ++) {
            ifs >> z[k] >> dummy;
        }
        int rsz, dsz = 1;

        ofstream ofs(opath, ios::binary);

        rsz = 2 * sizeof(int);
        record(ofs, rsz);
        ofs.write((char*)&dsz, sizeof(int));
        ofs.write((char*)&dsz, sizeof(int));
        record(ofs, rsz);

        rsz = 3 * sizeof(int);
        int xnum = imax - 2*gc*gc_mask;
        int ynum = jmax - 2*gc*gc_mask;
        int znum = kmax - 2*gc*gc_mask;
        record(ofs, rsz);
        ofs.write((char*)&xnum, sizeof(int));
        ofs.write((char*)&ynum, sizeof(int));
        ofs.write((char*)&znum, sizeof(int));
        record(ofs, rsz);

        rsz = sizeof(int) + sizeof(float);
        int   step = 0;
        float time = 0;
        record(ofs, rsz);
        ofs.write((char*)&step, sizeof(int));
        ofs.write((char*)&time, sizeof(float));
        record(ofs, rsz);

        size_t offset = gc*gc_mask;

        rsz = sizeof(float) * xnum;
        record(ofs, rsz);
        for (size_t i = 0; i < xnum; i ++) {
            ofs.write((char*)&x[i + offset], sizeof(float));
        }
        record(ofs, rsz);

        rsz = sizeof(float) * ynum;
        record(ofs, rsz);
        for (size_t j = 0; j < ynum; j ++) {
            ofs.write((char*)&y[j + offset], sizeof(float));
        }
        record(ofs, rsz);

        rsz = sizeof(float) * znum;
        record(ofs, rsz);
        for (size_t k = 0; k < znum; k ++) {
            ofs.write((char*)&z[k + offset], sizeof(float));
        }
        record(ofs, rsz);

        free(x);
        free(y);
        free(z);
        ofs.close();
    } else if (typeid(T) == typeid(double)) {
        double *x, *y, *z, dummy;
        x = (double*)malloc(sizeof(double)*imax);
        y = (double*)malloc(sizeof(double)*jmax);
        z = (double*)malloc(sizeof(double)*kmax);
        for (size_t i = 0; i < imax; i ++) {
            ifs >> x[i] >> dummy;
        }
        for (size_t j = 0; j < jmax; j ++) {
            ifs >> y[j] >> dummy;
        }
        for (size_t k = 0; k < kmax; k ++) {
            ifs >> z[k] >> dummy;
        }
        int rsz, dsz = 2;

        ofstream ofs(opath, ios::binary);

        rsz = 2 * sizeof(int);
        record(ofs, rsz);
        ofs.write((char*)&dsz, sizeof(int));
        ofs.write((char*)&dsz, sizeof(int));
        record(ofs, rsz);

        rsz = 3 * sizeof(int64_t);
        int64_t xnum = imax - 2*gc*gc_mask;
        int64_t ynum = jmax - 2*gc*gc_mask;
        int64_t znum = kmax - 2*gc*gc_mask;
        record(ofs, rsz);
        ofs.write((char*)&xnum, sizeof(int64_t));
        ofs.write((char*)&ynum, sizeof(int64_t));
        ofs.write((char*)&znum, sizeof(int64_t));
        record(ofs, rsz);

        rsz = sizeof(int64_t) + sizeof(double);
        int64_t step = 0;
        double  time = 0;
        record(ofs, rsz);
        ofs.write((char*)&step, sizeof(int64_t));
        ofs.write((char*)&time, sizeof(double));
        record(ofs, rsz);

        size_t offset = gc*gc_mask;

        rsz = sizeof(double) * xnum;
        record(ofs, rsz);
        for (size_t i = 0; i < xnum; i ++) {
            ofs.write((char*)&x[i + offset], sizeof(double));
        }
        record(ofs, rsz);

        rsz = sizeof(double) * ynum;
        record(ofs, rsz);
        for (size_t j = 0; j < ynum; j ++) {
            ofs.write((char*)&y[j + offset], sizeof(double));
        }
        record(ofs, rsz);

        rsz = sizeof(double) * znum;
        record(ofs, rsz);
        for (size_t k = 0; k < znum; k ++) {
            ofs.write((char*)&z[k + offset], sizeof(double));
        }
        record(ofs, rsz);

        free(x);
        free(y);
        free(z);
        ofs.close();
    } else {
        printf("undefined data type\n");
    }
    ifs.close();
}

int main(int argc, char **argv) {
    string prefix(argv[1]);
    printf("VisiFalm: converting %s files to Riken V-Isio files\n", prefix.c_str());
    read_index_file(prefix+".json");
    cvnode_to_crd<float>(prefix+".cv", prefix+".crd", true);
    uvwp_to_sph<float>(prefix, true);
}