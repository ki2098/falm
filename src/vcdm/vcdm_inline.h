#ifndef VCDM_VCDM_INLINE_H
#define VCDM_VCDM_INLINE_H


#include "vcdm_type.h"

namespace Vcdm {

inline void VcdmDomain::write(FILE *file) {
    crdFileEndian = getEndian();

    fprintf(file, "Domain {\n");
    fprintf(file, "  GlobalOrigin            = (%e, %e, %e)\n", globalOrigin[0]  , globalOrigin[1]  , globalOrigin[2]  );
    fprintf(file, "  GlobalRegion            = (%e, %e, %e)\n", globalRegion[0]  , globalRegion[1]  , globalRegion[2]  );
    fprintf(file, "  GlobalVoxel             = (%d, %d, %d)\n", globalVoxel[0]   , globalVoxel[1]   , globalVoxel[2]   );
    fprintf(file, "  GlobalDivision          = (%d, %d, %d)\n", globalDivision[0], globalDivision[1], globalDivision[2]);
    fprintf(file, "  ActiveSubdomainFile     = \"%s\"\n", activeSubdomainFile.c_str());
    fprintf(file, "  CoordinateFile          = \"%s\"\n", crdFile.c_str());
    fprintf(file, "  CoordinateFileType      = \"binary\"\n");

if (dtype == DataType::FLOAT64) {
    fprintf(file, "  CoordinateFilePrecision = \"Float64\"\n");
} else if (dtype == DataType::FLOAT32) {
    fprintf(file, "  CoordinateFilePrecision = \"Float32\"\n");
}

if (crdFileEndian == Endian::Little) {
    fprintf(file, "  CoordinateFileEndian    = \"little\"\n");
} else if (crdFileEndian == Endian::Big) {
    fprintf(file, "  CoordinateFileEndian    = \"Big\"\n");
}

    fprintf(file, "}\n\n");
}

inline void VcdmMPI::write(FILE *file) {
    fprintf(file, "MPI {\n");
    fprintf(file, "  NumberOfRank            = %d\n", size);
    fprintf(file, "  NumberOfGroup           = %d\n", ngrp);
    fprintf(file, "}\n\n");
}

inline void VcdmRank::write(FILE *file) {
    fprintf(file, "  Rank[@] {\n");
    fprintf(file, "    ID                    = %d\n", rank);
    fprintf(file, "    HostName              = \"%s\"\n", hostName.c_str());
    fprintf(file, "    VoxelSize             = (%d, %d, %d)\n", voxelSize[0], voxelSize[1], voxelSize[2]);
    fprintf(file, "    HeadIndex             = (%d, %d, %d)\n", headIdx[0], headIdx[1], headIdx[2]);
    fprintf(file, "    TailIndex             = (%d, %d, %d)\n", tailIdx[0], tailIdx[1], tailIdx[2]);
    fprintf(file, "  }\n");
}

inline void VcdmSlice::write(FILE *file) {
    fprintf(file, "  Slice[@] {\n");
    fprintf(file, "    Step              = %d\n", step);
    fprintf(file, "    Time              = %e\n", time);

if (!avgMode) {
    fprintf(file, "    AveragedStep      = %d\n", avgStep);
    fprintf(file, "    AveragedTime      = %e\n", avgTime);
}

if (varMin.size() == 3) {
    fprintf(file, "    VectorMinMax {\n");
    fprintf(file, "      Min             = %e\n", vectorMin);
    fprintf(file, "      Max             = %e\n", vectorMax);
    fprintf(file, "    }\n");
}

for (int i = 0; i < varMin.size(); i ++) {
    fprintf(file, "    MinMax[@] {\n");
    fprintf(file, "      Min             = %e\n", varMin[i]);
    fprintf(file, "      Max             = %e\n", varMax[i]);
    fprintf(file, "    }\n");
}

    fprintf(file, "  }\n");
}

inline void VcdmFinfo::write(FILE *file) {
    endian = getEndian();

    std::string _dirpath = dirPath + "/";

    fprintf(file, "FileInfo {\n");
    fprintf(file, "  DFIType             = \"%s\"\n", dfiType.c_str());
    fprintf(file, "  DirectoryPath       = \"%s\"\n", _dirpath.c_str());

if (timeSliceDir) {
    fprintf(file, "  TimeSliceDirectory  = \"on\"\n");
} else {
    fprintf(file, "  TimeSliceDirectory  = \"off\"\n");
}

    fprintf(file, "  Prefix              = \"%s\"\n", prefix.c_str());

if (fFormat == FileFormat::SPH) {
    fprintf(file, "  FileFormat          = \"sph\"\n");
} else if (fFormat == FileFormat::BOV) {
    fprintf(file, "  FileFormat          = \"bov\"\n");
} else if (fFormat == FileFormat::PLOT3D) {
    fprintf(file, "  FileFormat          = \"plot3d\"\n");
}

if (fnameFormat == FilenameFormat::RANK_STEP) {
    fprintf(file, "  FieldFilenameFormat = \"rank_step\"\n");
} else if (fnameFormat == FilenameFormat::STEP_RANK) {
    fprintf(file, "  FieldFilenameFormat = \"step_rank\"\n");
}

    fprintf(file, "  RankNoPrefix        = \"%s\"\n", rankPrefix.c_str());
    fprintf(file, "  GuideCell           = %d\n", gc);

if (dtype == DataType::FLOAT64) {
    fprintf(file, "  DataType            = \"Float64\"\n");
} else if (dtype == DataType::FLOAT32) {
    fprintf(file, "  DataType            = \"Float32\"\n");
}

if (endian == Endian::Little) {
    fprintf(file, "  Endian              = \"little\"\n");
} else if (endian == Endian::Big) {
    fprintf(file, "  Endian              = \"big\"\n");
}

    fprintf(file, "  NumVariables        = %d\n", varList.size());

for (int i = 0; i < varList.size(); i ++) {
    fprintf(file, "  Variable[@] { name  = \"%s\" }\n", varList[i].c_str());
}

    fprintf(file, "}\n\n");
}

template<typename T>
void VCDM<T>::writeProcess(FILE *file) {
    fprintf(file, "Process {\n");

for (int i = 0; i < dfiProc.size(); i ++) {
    dfiProc[i].write(file);
}

    fprintf(file, "}\n\n");
}

template<typename T>
void VCDM<T>::writeFilePath(FILE *file) {
    std::string procdfiname = dfiFinfo.prefix + procSuffix;
    fprintf(file, "FilePath {\n");
    fprintf(file, "  Process             = \"%s\"\n", procdfiname.c_str());
    fprintf(file, "}\n\n");
}

template<typename T>
void VCDM<T>::writeTimeSlice(FILE *file) {
    fprintf(file, "TimeSlice {\n");
for (int i = 0; i < timeSlice.size(); i ++) {
    timeSlice[i].write(file);
}
    fprintf(file, "}\n\n");
}

template<typename T>
void VCDM<T>::writeUnitList(FILE *file) {
    fprintf(file, "UnitList {\n");
    fprintf(file, "  Length {\n");
    fprintf(file, "    Unit              = \"NonDimensional\"\n");
    fprintf(file, "    Reference         = 1.0\n");
    fprintf(file, "  }\n");
    fprintf(file, "  Velocity {\n");
    fprintf(file, "    Unit              = \"NonDimensional\"\n");
    fprintf(file, "    Reference         = 1.0\n");
    fprintf(file, "  }\n");
    fprintf(file, "  Pressure {\n");
    fprintf(file, "    Unit              = \"NonDimensional\"\n");
    fprintf(file, "    Reference         = 0.0\n");
    fprintf(file, "    Difference        = 1.0\n");
    fprintf(file, "  }\n");
    fprintf(file, "  Temperature {\n");
    fprintf(file, "    Unit              = \"NonDimensional\"\n");
    fprintf(file, "    Reference         = 0.0\n");
    fprintf(file, "    Difference        = 1.0\n");
    fprintf(file, "  }\n");
    fprintf(file, "}\n\n");
}

template<typename T>
void VCDM<T>::writeProcDfi() {
    std::string filename = outputDir + "/" + dfiFinfo.prefix + procSuffix;
    FILE *file = fopen(filename.c_str(), "w");
    dfiDomain.write(file);
    dfiMPI.write(file);
    writeProcess(file);
    fclose(file);
}

template<typename T>
void VCDM<T>::writeIndexDfi() {
    std::string filename = outputDir + "/" + dfiFinfo.prefix + indexSuffix;
    FILE *file = fopen(filename.c_str(), "w");
    dfiFinfo.write(file);
    writeFilePath(file);
    writeUnitList(file);
    writeTimeSlice(file);
    fclose(file);
}

template<typename T>
std::string VCDM<T>::makeFilename(
    FilenameFormat fnameFormat,
    std::string    prefix,
    std::string    rankPrefix,
    std::string    ext,
    int            rank,
    int            step
) {
    size_t len = prefix.size() + rankPrefix.size() + ext.size() + 100;
    char *tmp = (char *)malloc(sizeof(char) * len);

    if(fnameFormat == FilenameFormat::RANK) {
        sprintf(tmp, "%s%s%06d.%s", prefix.c_str(), rankPrefix.c_str(), rank, ext.c_str());
    } else if (fnameFormat == FilenameFormat::STEP_RANK) {
        sprintf(tmp, "%s_%010d%s%06d.%s", prefix.c_str(), step, rankPrefix.c_str(), rank, ext.c_str());
    } else if (fnameFormat == FilenameFormat::RANK_STEP) {
        sprintf(tmp, "%s%s%06d_%010d.%s", prefix.c_str(), rankPrefix.c_str(), rank, step, ext.c_str());
    }

    std::string fname(tmp);
    free(tmp);
    return fname;
}

template<typename T>
void VCDM<T>::writeXYZ(T *xyz, int gc, int rank, int step, IdxType idxtype) {
    std::string fname = outputDir + "/" + makeFilename(
        FilenameFormat::RANK,
        dfiFinfo.prefix,
        dfiFinfo.rankPrefix,
        "xyz",
        rank,
        step
    );

    FILE *file = fopen(fname.c_str(), "wb");

    int3 size = dfiProc[rank].voxelSize;
    size[0] += 2 * gc;
    size[1] += 2 * gc;
    size[2] += 2 * gc;
    int sz3d_gc = size[0] * size[1] * size[2];
    int rsz;

    rsz = sizeof(int) * 3;
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&size[0], sizeof(int), 1, file);
    fwrite(&size[1], sizeof(int), 1, file);
    fwrite(&size[2], sizeof(int), 1, file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sz3d_gc * 3 * sizeof(T);
    fwrite(&rsz, sizeof(int), 1, file);
    if (idxtype == IdxType::IJKN) {
        fwrite(xyz, sizeof(T), sz3d_gc * 3, file);
    } else if (idxtype == IdxType::NIJK) {
        for (int n = 0; n < 3; n ++) {
        for (int k = 0; k < size[2]; k ++) {
        for (int j = 0; j < size[1]; j ++) {
        for (int i = 0; i < size[0]; i ++) {
            fwrite(&xyz[NIJK_IDX(n, i, j, k, size, 3)], sizeof(T), 1, file);
        }}}}
    }
    fwrite(&rsz, sizeof(int), 1, file);

    fclose(file);
}

template<typename T>
void VCDM<T>::writeFunc(T *data, int gc, int dim, int rank, int step, IdxType idxtype) {
    std::string fname = outputDir + "/" + makeFilename(
        dfiFinfo.fnameFormat,
        dfiFinfo.prefix,
        dfiFinfo.rankPrefix,
        "fun",
        rank,
        step
    );

    FILE *file = fopen(fname.c_str(), "wb");

    int3 size = dfiProc[rank].voxelSize;
    size[0] += 2 * gc;
    size[1] += 2 * gc;
    size[2] += 2 * gc;
    int sz3d_gc = size[0] * size[1] * size[2];
    int rsz;

    rsz = sizeof(int) * 4;
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&size[0], sizeof(int), 1, file);
    fwrite(&size[1], sizeof(int), 1, file);
    fwrite(&size[2], sizeof(int), 1, file);
    fwrite(&dim, sizeof(int), 1, file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(T) * sz3d_gc * dim;
    fwrite(&rsz, sizeof(int), 1, file);
    if (idxtype == IdxType::IJK || idxtype == IdxType::IJKN) {
        fwrite(data, sizeof(T), sz3d_gc * dim, file);
    } else {
        for (int n = 0; n < dim; n ++) {
        for (int k = 0; k < size[2]; k ++) {
        for (int j = 0; j < size[1]; j ++) {
        for (int i = 0; i < size[0]; i ++) {
            fwrite(&data[NIJK_IDX(n, i, j, k,size, dim)], sizeof(T), 1, file);
        }}}}
    }
    fwrite(&rsz, sizeof(int), 1, file);

    fclose(file);
}

template<typename T>
void Vcdm::VCDM<T>::writeCrd(T *x, T *y, T *z, int gc) {
    std::string filename = outputDir + "/" + dfiFinfo.prefix + ".crd";
    FILE *file = fopen(filename.c_str(), "wb");
    int3 size = dfiDomain.globalVoxel;
    int rsz, dummy;
    float fdummy;

    rsz = sizeof(int) * 2;
    dummy = 1;
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&dummy, sizeof(int), 1, file);
    fwrite(&dummy, sizeof(int), 1, file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(int) * 3;
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&size[0], sizeof(int), 1, file);
    fwrite(&size[1], sizeof(int), 1, file);
    fwrite(&size[2], sizeof(int), 1, file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(int) + sizeof(float);
    dummy = 0;
    fdummy = 0;
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&dummy, sizeof(float), 1, file);
    fwrite(&fdummy, sizeof(int), 1, file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(T) * size[0];
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&x[gc], sizeof(T), size[0], file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(T) * size[1];
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&y[gc], sizeof(T), size[1], file);
    fwrite(&rsz, sizeof(int), 1, file);

    rsz = sizeof(T) * size[2];
    fwrite(&rsz, sizeof(int), 1, file);
    fwrite(&z[gc], sizeof(T), size[2], file);
    fwrite(&rsz, sizeof(int), 1, file);

    fclose(file);
}

}

#endif
