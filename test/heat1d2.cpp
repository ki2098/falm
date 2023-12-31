#include <stdio.h>
#include "../src/FalmEq.h"
#include "../src/MV.h"

using namespace Falm;

#define USE_CUDA_AWARE_MPI true

#define Nx 100
#define Ny 1
#define Nz 1
#define Lx 1.0
#define TW 0.0
#define TE 100.0

void set_matrix_value(Matrix<REAL> &x, INT3 range_shape, INT3 range_offset, INT3 pshape, REAL value) {
    for (INT i = 0; i < range_shape[0]; i ++) {
        for (INT j = 0; j < range_shape[1]; j ++) {
            for (INT k = 0; k < range_shape[2]; k ++) {
                INT _i = i + range_offset[0];
                INT _j = j + range_offset[1];
                INT _k = k + range_offset[2];
                x(IDX(_i, _j, _k, pshape)) = value;
            }
        }
    }
}

void print_eq(Matrix<REAL> &a, Matrix<REAL> &b, INT3 shape) {
    printf("%s = %s\n", a.cname(), b.cname());
    for (INT i = Gd; i < shape[0] - Gd; i ++) {
        for (INT j = Gd; j < shape[1] - Gd; j ++) {
            for (INT k = Gd; k < shape[2] - Gd; k ++) {
                INT idx = IDX(i, j, k, shape);
                for (INT m = 0; m < a.shape[1]; m ++) {
                    printf("%12lf ", a(idx, m));
                }
                printf("= %12lf\n", b(idx));
            }
        }
    }
}

void print_result(Matrix<REAL> &x, Matrix<REAL> &r, INT3 shape) {
    for (INT i = Gd; i < shape[0] - Gd; i ++) {
        for (INT j = Gd; j < shape[1] - Gd; j ++) {
            for (INT k = Gd; k < shape[2] - Gd; k ++) {
                INT idx = IDX(i, j, k, shape);
                printf("%12.4lf %12.4lf\n", x(idx), r(idx));
            }
        }
    }
}

INT dim_division(INT dim_size, INT mpi_size, INT mpi_rank) {
    INT p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

void print_xy_slice(Matrix<REAL> &x, INT3 domain_shape, INT slice_at_z) {
    for (INT j = domain_shape[1] - 1; j >= 0; j --) {
        for (INT i = 0; i < domain_shape[0]; i ++) {
            REAL value = x(IDX(i, j, slice_at_z, domain_shape));
            if (value == 0) {
                printf(".   ", value);
            } else {
                printf("%-3.0lf ", value);
            }
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);

    Region global(
        INT3{Nx + 2 * Gd, Ny + 2 * Gd, Nz + 2 * Gd},
        INT3{0, 0, 0}
    );
    Matrix<REAL> ga, gt, gb, gr;
    ga.alloc(global.shape, 7, HDC::Host  , "global a");
    gt.alloc(global.shape, 1, HDC::Device, "global t");
    gb.alloc(global.shape, 1, HDC::Host  , "global b");
    gr.alloc(global.shape, 1, HDC::Device, "global r");
    const REAL dx = Lx / Nx;
    for (INT i = Gd; i < Gd + Nx; i ++) {
        REAL ac, ae, aw, bc;
        if (i == Gd) {
            ae = 1.0 / (dx * dx);
            aw = 0.0;
            ac = - (ae + 2.0 / (dx * dx));
            bc = - (2 * TW) / (dx * dx);
        } else if (i == Gd + Nx - 1) {
            ae = 0.0;
            aw = 1.0 / (dx * dx);
            ac = - (aw + 2.0 / (dx * dx));
            bc = - (2 * TE) / (dx * dx);
        } else {
            ae = 1.0 / (dx * dx);
            aw = 1.0 / (dx * dx);
            ac = - (ae + aw);
            bc = 0.0;
        }
        INT idx = IDX(i, Gd, Gd, global.shape);
        ga(idx, 0) = ac;
        ga(idx, 1) = ae;
        ga(idx, 2) = aw;
        gb(idx)    = bc;
    }

    CPM cpm;
    cpm.use_cuda_aware_mpi = USE_CUDA_AWARE_MPI;
    printf("using cuda aware mpi: %d \n");
    CPM_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPM_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = {cpm.size, 1, 1};
    cpm.initNeighbour();
    printf("%d(%u %u %u): E%2d W%2d N%2d S%2d T%2d B%2d\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2], cpm.neighbour[0], cpm.neighbour[1], cpm.neighbour[2], cpm.neighbour[3], cpm.neighbour[4], cpm.neighbour[5]);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(cpm.rank % gpu_count);
    printf("process %d running no device %d\n", cpm.rank, cpm.rank % gpu_count);
    CPM_Barrier(MPI_COMM_WORLD);

    INT ox = 0, oy = 0, oz = 0;
    for (INT i = 0; i < cpm.idx[0]; i ++) {
        ox += dim_division(Nx, cpm.shape[0], i);
    }
    for (INT j = 0; j < cpm.idx[1]; j ++) {
        oy += dim_division(Ny, cpm.shape[1], j);
    }
    for (INT k = 0; k < cpm.idx[2]; k ++) {
        oz += dim_division(Nz, cpm.shape[2], k);
    }
    Region process(
        INT3{dim_division(Nx, cpm.shape[0], cpm.idx[0]) + Gdx2, dim_division(Ny, cpm.shape[1], cpm.idx[1]) + Gdx2, dim_division(Nz, cpm.shape[2], cpm.idx[2]) + Gdx2},
        INT3{ox, oy, oz}
    );
    for (INT i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            printf("%-2d(%-2u %-2u %-2u): (%-3u %-3u %-3u) (%-3u %-3u %-3u)\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2], process.shape[0], process.shape[1], process.shape[2], process.offset[0], process.offset[1], process.offset[2]);
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    Matrix<REAL> a, t, b, r;
    a.alloc(process.shape, 7, HDC::Host  , "a");
    t.alloc(process.shape, 1, HDC::Device, "t");
    b.alloc(process.shape, 1, HDC::Host  , "b");
    r.alloc(process.shape, 1, HDC::Device, "r");
    for (INT i = Gd; i < process.shape[0] - Gd; i ++) {
        for (INT j = Gd; j < process.shape[1] - Gd; j ++) {
            for (INT k = Gd; k < process.shape[2] - Gd; k ++) {
                INT gi = i + process.offset[0];
                INT gj = j + process.offset[1];
                INT gk = k + process.offset[2];
                INT idx = IDX(i, j, k, process.shape);
                INT gidx = IDX(gi, gj, gk, global.shape);
                for (INT m = 0; m < 7; m ++) {
                    a(idx, m) = ga(gidx, m);
                }
                b(idx) = gb(gidx);
            }
        }
    }

    INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(process, Gd));
    Matrix<REAL> region(process.shape, 1, HDC::Host, "region");
    set_matrix_value(region, inner_shape, inner_offset, process.shape, cpm.rank * 10);
    for (INT i = 0; i < 6; i ++) {
        if (cpm.neighbour[i] >= 0) {
            set_matrix_value(region, boundary_shape[i], boundary_offset[i], process.shape, 100 + i);
        }
    }

    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            print_xy_slice(region, process.shape, process.shape[2] / 2);
            printf("\n");
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    a.sync(MCP::Hst2Dev);
    b.sync(MCP::Hst2Dev);
    dim3 block_dim(32, 1, 1);
    REAL max_diag = L2Dev_MaxDiag(a, process, block_dim, cpm);
    printf("%12lf\n", max_diag);
    L1Dev_ScaleMatrix(a, 1.0 / max_diag, block_dim);
    L1Dev_ScaleMatrix(b, 1.0 / max_diag, block_dim);
    a.sync(MCP::Dev2Hst);
    b.sync(MCP::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            print_eq(a, b, process.shape);
            printf("\n");
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }
    STREAM faceStream[6];
    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamCreate(&faceStream[fid]);
    }
    FalmEq solver(LSType::PBiCGStab, 1000, 1e-9, 1.2, LSType::SOR, 5, 1.5);
    solver.Solve(a, t, b, r, global, process, block_dim, cpm, faceStream);
    t.sync(MCP::Dev2Hst);
    r.sync(MCP::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            print_result(t, r, process.shape);
            printf("\n");
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }
    printf("%d %.12lf\n", solver.it, solver.err);
    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamDestroy(faceStream[fid]);
    }

    return CPM_Finalize();
}