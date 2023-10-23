#ifndef VCDM_VCDM_TYPE_H
#define VCDM_VCDM_TYPE_H

#include <cassert>
#include <string>
#include <string.h>
#include <vector>
#include <stdio.h>

namespace Vcdm {

struct intx3 {int x, y, z;};

struct doublex3 {double x, y, z;};

enum class FilenameFormat {RANK, STEP_RANK, RANK_STEP};
enum class FileFormat {SPH, BOV, PLOT3D};
enum class DFIType {Cartesian, NonUniformCartesian};
enum class Endian {Big, Little};
enum class IdxType {IJK, NIJK, IJKN};
enum class DataType {INT32, INT64, FLOAT32, FLOAT64};

static inline Endian getEndian() {
    const int32_t i = 0x0001;
    const char* b = (const char*)(&i);
    if (b[0]) {
        return Endian::Little;
    } else {
        return Endian::Big;
    }
}

struct VcdmDomain {
    doublex3       globalOrigin;
    doublex3       globalRegion;
    intx3          globalVoxel;
    intx3          globalDivision;
    DataType       dtype;

    void write(FILE *file);

    std::string activeSubdomainFile = "";
    std::string crdFile             = "";
    std::string crdFileType         = "binary";
    Endian      crdFileEndian;
};

struct VcdmMPI {
    int size;

    void write(FILE *file);
    
    int ngrp = 1;
};

struct VcdmRank {
    int         rank;
    intx3       voxelSize;
    intx3       headIdx;
    intx3       tailIdx;

    void write(FILE *file);

    std::string hostName = "";
};

struct VcdmFinfo {
    std::string              prefix;
    int                      gc;
    std::vector<std::string> varList;
    std::string              rankPrefix   = "_id";
    DataType                 dtype;

    void write(FILE *file);

    std::string              dfiType      = "Non_Uniform_Cartesian";
    bool                     timeSliceDir = false;
    FileFormat               fFormat      = FileFormat::PLOT3D;
    FilenameFormat           fnameFormat  = FilenameFormat::STEP_RANK;
    Endian                   endian;
    std::string              dirPath      = ".";
};

struct VcdmSlice {
    int                 step;
    double              time;
    int                 avgStep;
    double              avgTime;
    double              vectorMin;
    double              vectorMax;
    std::vector<double> varMin;
    std::vector<double> varMax;
    bool                avgMode;

    void write(FILE *file);
};

template<typename T>
class VCDM {
public:
    VcdmDomain             dfiDomain;
    VcdmMPI                dfiMPI;
    VcdmFinfo              dfiFinfo;
    std::vector<VcdmRank>  dfiProc;
    std::vector<VcdmSlice> timeSlice;

    void setPath(const std::string &_dir, const std::string &_prefix) {
        outputDir = _dir + "/" + cut_dirpath(_prefix);
        dfiFinfo.prefix = cut_filename(_prefix);
    }
    
    void writeProcDfi();
    void writeIndexDfi();

    std::string makeFilename(
        FilenameFormat fnameFormat,
        std::string    prefix,
        std::string    rankPrefix,
        std::string    ext,
        int            rank,
        int            step
    );

    std::string makeFileDataExt() {
        FileFormat &fformat = dfiFinfo.fFormat;
        if (fformat == FileFormat::PLOT3D) {
            return "fun";
        } else if (fformat == FileFormat::SPH) {
            return "sph";
        } else if (fformat == FileFormat::BOV) {
            return "bov";
        }
        return "";
    }

    VCDM() {
        if (typeid(T) == typeid(double)) {
            dfiDomain.dtype = dfiFinfo.dtype = DataType::FLOAT64;
        } else if (typeid(T) == typeid(float)) {
            dfiDomain.dtype = dfiFinfo.dtype = DataType::FLOAT32;
        }
    }

    void writeFileData(T *data, int gc, int dim, int rank, int step, IdxType idxtype) {
        if (dfiFinfo.fFormat == FileFormat::PLOT3D) {
            writeFunc(data, gc, dim, rank, step, idxtype);
        }
    }

    void writeGridData(T *xyz, int gc, int rank, int step, IdxType idxtype) {
        if (dfiFinfo.fFormat == FileFormat::PLOT3D) {
            writeXYZ(xyz, gc, rank, step, idxtype);
        }
    }

protected:
    std::string           indexSuffix = "_index.dfi";
    std::string           procSuffix  = "_proc.dfi";
    std::string           outputDir;

    void writeProcess(FILE *file);
    
    std::string cut_filename(const std::string &path) {
        if (path.find_last_of('/') == std::string::npos) {
            return path;
        } else {
            return path.substr(path.find_last_of('/') + 1);
        }
    }

    std::string cut_dirpath(const std::string &path) {
        if (path.find_first_of('/') == std::string::npos) {
            return "";
        } else {
            return path.substr(0, path.find_last_of('/'));
        }
    }

    void writeFilePath(FILE *file);

    void writeUnitList(FILE *file);

    void writeTimeSlice(FILE *file);

    int IJK_IDX(int i, int j, int k, const intx3 &size) {
        return i + j * size.x + k * size.x * size.y;
    }

    int IJKN_IDX(int i, int j, int k, int n, const intx3 &size) {
        return i + j * size.x + k * size.x * size.y + n * size.x * size.y * size.z;
    }

    int NIJK_IDX(int n, int i, int j, int k, const intx3 &size, int dim) {
        return n + i * dim + j * dim * size.x + k * dim * size.x * size.y;
    }

    void writeXYZ(T *xyz, int gc, int rank, int step, IdxType idxtype);

    void writeFunc(T *data, int gc, int dim, int rank, int step, IdxType idxtype);

};

}

#endif