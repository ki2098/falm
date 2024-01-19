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

    size4(const size3 &sz3, size_t n) {
        _sz[0] = sz3[0];
        _sz[1] = sz3[1];
        _sz[2] = sz3[2];
        _sz[3] = n;
    }

    size_t product() const {
        return _sz[0] * _sz[1] * _sz[2] * _sz[3];
    }

    size_t idx(size_t i, size_t j, size_t k, size_t n) {
        return i + j * _sz[0] + k * _sz[0] * _sz[1] + n * _sz[0] * _sz[1] * _sz[2];
    }
};

struct TimeSliceInfo {
    size_t step;
    double time;
    bool tavg;
};

vector<size3> size_list;
vector<size3> offset_list;
size3 global;
size_t gc;
vector<TimeSliceInfo> slice_list;
int mpi_size;
size_t dtype;
size_t n_variable;

string make_proc_filename(string prefix, int rank, size_t step) {
    char *tmp = (char*)malloc(sizeof(char) * (prefix.size() + 32));
    sprintf(tmp, "%s_%06d_%010d", prefix.c_str(), rank, step);
    return string(tmp);
}

string make_filename(string prefix, size_t step) {
    char *tmp = (char*)malloc(sizeof(char) * (prefix.size() + 32));
    sprintf(tmp, "%s_%010d", prefix.c_str(), step);
    return string(tmp);
}

void readIndexFile(string path) {
    ifstream idxfile(path);
    json idxjson = json::parse(idxfile);
    auto jglobal = idxjson["global"];
    gc = idxjson["guidePoint"].get<size_t>();
    global = size3(jglobal) + 2*gc;

    auto rksj = idxjson["ranks"];
    mpi_size = rksj.size();
    size_list = vector<size3>(mpi_size);
    offset_list = vector<size3>(mpi_size);
    for (auto rj : rksj) {
        int rank = rj["rank"].get<int>();
        offset_list[rank] = size3(rj["offset"]);
        size_list[rank] = size3(size3(rj["voxel"]) + 2*gc);
    }
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
    if (idxjson["dataType"].get<string>() == "float32") {
        dtype = sizeof(float);
    } else if (idxjson["dataType"].get<string>() == "float64") {
        dtype = sizeof(double);
    } else {
        dtype = sizeof(int);
    }
    n_variable = idxjson["variables"].size();
}

template<typename T>
void reconstruct(std::string prefix, size_t step, double time) {
    T *v = (T *)malloc(sizeof(T) * global.product() * n_variable);
    printf("reconstructing %s with maximum %d threads...\n", make_filename(prefix, step).c_str(), omp_get_max_threads());
    fflush(stdout);
    for (int rank = 0; rank < mpi_size; rank ++) {
        size3 size = size_list[rank];
        size3 offset = offset_list[rank];
        size3 start(0, 0, 0);
        for (int i = 0; i < 3; i ++) {
            if (offset[i] != 0) {
                start[i] = gc;
            }
        }
        size3 end = size;
        for (int i = 0; i < 3; i ++) {
            if (offset[i] + size[i] != global[i]) {
                end[i] -= gc;
            }
        }
        
        T *vp = (T*)malloc(sizeof(T) * size.product() * n_variable);
        string procfilename = make_proc_filename(prefix, rank, step);
        ifstream procfile(procfilename, ios::binary);
        size_t imax, jmax, kmax, nvar, tstep, dsz, pgc;
        double ttime;
        procfile.read((char*)&imax, sizeof(size_t));
        procfile.read((char*)&jmax, sizeof(size_t));
        procfile.read((char*)&kmax, sizeof(size_t));
        procfile.read((char*)&nvar, sizeof(size_t));
        procfile.read((char*)&pgc, sizeof(size_t));
        procfile.read((char*)&tstep, sizeof(size_t));
        procfile.read((char*)&ttime, sizeof(double));
        procfile.read((char*)&dsz, sizeof(size_t));

        assert(imax == size[0] && jmax == size[1] && kmax == size[2] && pgc == gc && nvar == n_variable && tstep == step && dsz == dtype);
        procfile.read((char*)vp, sizeof(T) * size.product() * n_variable);

        size4 vsz(size, n_variable);
        size4 gvsz(global, n_variable);
        #pragma omp parallel for collapse(4)
        for (size_t n = 0; n < n_variable; n ++) {
        for (size_t k = start[2]; k < end[2]; k ++) {
        for (size_t j = start[1]; j < end[1]; j ++) {
        for (size_t i = start[0]; i < end[0]; i ++) {
            size_t gi = i + offset[0];
            size_t gj = j + offset[1];
            size_t gk = k + offset[2];
            v[gvsz.idx(gi, gj, gk, n)] = vp[vsz.idx(i, j, k, n)];
        }}}}

        procfile.close();
    }

    string ofilename = make_filename(prefix, step);
    ofstream ofile(ofilename, ios::binary);
    ofile.write((char*)&global[0], sizeof(size_t));
    ofile.write((char*)&global[1], sizeof(size_t));
    ofile.write((char*)&global[2], sizeof(size_t));
    ofile.write((char*)&n_variable, sizeof(size_t));
    ofile.write((char*)&gc, sizeof(size_t));
    ofile.write((char*)&step, sizeof(size_t));
    ofile.write((char*)&time, sizeof(double));
    ofile.write((char*)&dtype, sizeof(size_t));
    ofile.write((char*)v, sizeof(T) * global.product() * n_variable);
    ofile.close();
}

int main(int argc, char **argv) {
    std::string path(argv[1]);
    readIndexFile(path + ".json");
    std::string tavg_path = path + "_tavg";
    for (auto slice : slice_list) {
        size_t _step = slice.step;
        double _time = slice.time;
        if (dtype == sizeof(double)) {
            reconstruct<double>(path, _step, _time);
            if (slice.tavg) {
                reconstruct<double>(tavg_path, _step, _time);
            }
        }  else if (dtype == sizeof(float)) {
            reconstruct<float>(path, _step, _time);
            if (slice.tavg) {
                reconstruct<float>(tavg_path, _step, _time);
            }
        } else {
            printf("undefined data width %lu\n", dtype);
        }
    }
    return 0;
}
