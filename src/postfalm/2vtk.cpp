#include <fstream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <string>
#include <vtkNew.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedIntArray.h>
#include "../nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

size_t imax, jmax, kmax, nmax, gc, step, type;
double tt, dummy;

vtkNew<vtkRectilinearGrid> grid;
vtkNew<vtkXMLRectilinearGridWriter> writer;
vtkNew<vtkFloatArray> xvtk, yvtk, zvtk, uvtk, pvtk, qvtk;

size_t id(size_t i, size_t j, size_t k, size_t n) {
    return n*imax*jmax*kmax + k*imax*jmax + j*imax + i;
}

size_t id(size_t i, size_t j, size_t k) {
    return k*imax*jmax + j*imax + i;
}

size_t vid(size_t i, size_t j, size_t k, size_t n, size_t n_size) {
    return k*imax*jmax*n_size + j*imax*n_size + i*n_size + n;
}

double *x, *y, *z;
string prefix;
string suffix;
bool is_tavg = false;

void convert_mesh() {
    string cvname = prefix + ".cv";
    ifstream ifs(cvname);
    ifs >> imax >> jmax >> kmax >> gc;
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
    ifs.close();

    grid->SetDimensions(imax, jmax, kmax);

    xvtk->SetNumberOfValues(imax);
    for (size_t i = 0; i < imax; i ++) {
        xvtk->SetValue(i, x[i]);
    }
    grid->SetXCoordinates(xvtk);

    yvtk->SetNumberOfValues(jmax);
    for (size_t j = 0; j < jmax; j ++) {
        yvtk->SetValue(j, y[j]);
    }
    grid->SetYCoordinates(yvtk);

    zvtk->SetNumberOfValues(kmax);
    for (size_t k = 0; k < kmax; k ++) {
        zvtk->SetValue(k, z[k]);
    }
    grid->SetZCoordinates(zvtk);
    cout << "read grid data from " << cvname << endl;

    uvtk->SetNumberOfComponents(3);
    uvtk->SetNumberOfTuples(imax*jmax*kmax);
    uvtk->SetName("u");
    pvtk->SetNumberOfComponents(1);
    pvtk->SetNumberOfTuples(imax*jmax*kmax);
    pvtk->SetName("p");
    qvtk->SetNumberOfComponents(1);
    qvtk->SetNumberOfTuples(imax*jmax*kmax);
    qvtk->SetName("q");
}

void convert_data() {
    string middle_name = "_";
    if (is_tavg) {
        middle_name += "tavg_";
    }
    string fname = prefix + middle_name + suffix;
    FILE *file = fopen(fname.c_str(), "rb");
    fread(&imax, sizeof(size_t), 1, file);
    fread(&jmax, sizeof(size_t), 1, file);
    fread(&kmax, sizeof(size_t), 1, file);
    fread(&nmax, sizeof(size_t), 1, file);
    fread(&gc  , sizeof(size_t), 1, file);
    fread(&step, sizeof(size_t), 1, file);
    fread(&tt  , sizeof(double), 1, file);
    fread(&type, sizeof(size_t), 1, file);

    double *data = new double[imax*jmax*kmax*nmax];
    fread(data, sizeof(double), imax*jmax*kmax*nmax, file);

    cout << "read uvwp data from " << fname << endl;
    fclose(file);

    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
    for (size_t d = 0; d < 3; d ++) {
        uvtk->SetValue(vid(i,j,k,d,3), data[id(i,j,k,d)]);
    }}}}

    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        pvtk->SetValue(vid(i,j,k,0,1), data[id(i,j,k,3)]);
    }}}

    
    grid->GetPointData()->AddArray(pvtk);
    grid->GetPointData()->AddArray(uvtk);

    if (!is_tavg) {

    grid->GetPointData()->AddArray(qvtk);
    double *q = new double[imax*jmax*kmax];
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
    cout << "complete calculating Q" << endl;

    
    for (size_t k = 0; k < kmax; k ++) {
    for (size_t j = 0; j < jmax; j ++) {
    for (size_t i = 0; i < imax; i ++) {
        qvtk->SetValue(vid(i,j,k,0,1), q[id(i,j,k)]);
    }}}

    delete[] q;
    }

    writer->SetFileName((fname + ".vtr").c_str());
    writer->Write();
    cout << "write data to " << fname + ".vtr" << endl;

    while (grid->GetPointData()->GetNumberOfArrays() > 0) {
        grid->GetPointData()->RemoveArray(0);
    }

    delete[] data;
}

int main(int argc, char **argv) {
    writer->SetCompressionLevel(1);
    writer->SetInputData(grid);
    prefix = string(argv[1]);
    convert_mesh();
    
    string idx_path = prefix + ".json";
    ifstream idx_file(idx_path);
    auto idx_json = json::parse(idx_file);

    if (argc == 2) {
        for (auto slice : idx_json["outputSteps"]) {
            int step = slice["step"];
            char buf[11] = {0};
            sprintf(buf, "%010d", step);
            suffix = string(buf);
            is_tavg = false;
            convert_data();

            if (slice.contains("timeAvg")) {
                is_tavg = true;
                convert_data();
            }
        }
    } else {
        int step = atoi(argv[2]);
        for (auto slice : idx_json["outputSteps"]) {
            if (step == slice["step"].get<int>()) {
                char buf[11] = {0};
                sprintf(buf, "%010d", step);
                suffix = string(buf);
                is_tavg = false;
                convert_data();

                if (slice.contains("timeAvg")) {
                    is_tavg = true;
                    convert_data();
                }
            }
        }
    }

    return 0;
}
