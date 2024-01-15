#include <fstream>
#include "../nlohmann/json.hpp"

using json = nlohmann::json;
using namespace std;

vector<pair<size_t, double> > slice_list;

void read_index_file(string path) {
    ifstream idxfile(path);
    json idxjson = json::parse(idxfile);
    for (auto slice : idxjson["outputSteps"]) {
        size_t _step = slice[0].get<size_t>();
        double _time = slice[1].get<double>();
        pair<size_t, double> pt{_step, _time};
        slice_list.push_back(pt);
    }
}

void record(ofstream &ofs, int sz) {
    ofs.write((char*)&sz, sizeof(int));
}

string make_filename(string prefix, size_t step) {
    char *tmp = (char*)malloc(sizeof(char) * (prefix.size() + 32));
    sprintf(tmp, "%s_%010d", prefix.c_str(), step);
    return string(tmp);
}

template<typename T>
void write_sph(string path, size_t imax, size_t jmax, size_t kmax, size_t nv, size_t step, double time) {
    
}

template<typename T>
void uvwp_to_sph(string prefix, size_t step, bool cut_gc) {
    for (auto slice : slice_list) {
        string filename = make_filename(prefix, step);
        ifstream ifile(filename, ios::binary);
        size_t imax, jmax, kmax, nv, gc, stp;
        double time;

        
    }
}

template<typename T>
void cvnode_to_crd(string ipath, string opath, bool cut_gc) {
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
    cvnode_to_crd<float>(prefix+".cv", prefix+".crd", true);
}