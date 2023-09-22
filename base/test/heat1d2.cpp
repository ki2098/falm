#include <stdio.h>
#include "../src/structEqL2.h"
#include "../src/MVL2.h"

using namespace Falm;

#define Nx 100
#define Ny 1
#define Nz 1
#define Lx 1.0
#define TW 0.0
#define TE 100.0

void set_matrix_value(Matrix<double> &x, uint3 range_shape, uint3 range_offset, uint3 pshape, double value) {
    for (unsigned int i = 0; i < range_shape.x; i ++) {
        for (unsigned int j = 0; j < range_shape.y; j ++) {
            for (unsigned int k = 0; k < range_shape.z; k ++) {
                unsigned int _i = i + range_offset.x;
                unsigned int _j = j + range_offset.y;
                unsigned int _k = k + range_offset.z;
                x(IDX(_i, _j, _k, pshape)) = value;
            }
        }
    }
}

void print_eq(Matrix<double> &a, Matrix<double> &b, uint3 shape) {
    for (unsigned int i = Gd; i < shape.x - Gd; i ++) {
        for (unsigned int j = Gd; j < shape.y - Gd; j ++) {
            for (unsigned int k = Gd; k < shape.z - Gd; k ++) {
                unsigned int idx = IDX(i, j, k, shape);
                for (unsigned int m = 0; m < a.shape.y; m ++) {
                    printf("%12lf ", a(idx, m));
                }
                printf("= %12lf\n", b(idx));
            }
        }
    }
}

void print_result(Matrix<double> &x, Matrix<double> &r, uint3 shape) {
    for (unsigned int i = Gd - 1; i < shape.x - Gd + 1; i ++) {
        for (unsigned int j = Gd; j < shape.y - Gd; j ++) {
            for (unsigned int k = Gd; k < shape.z - Gd; k ++) {
                unsigned int idx = IDX(i, j, k, shape);
                printf("%12.4lf %12.4lf\n", x(idx), r(idx));
            }
        }
    }
}

unsigned int dim_division(unsigned int dim_size, int mpi_size, int mpi_rank) {
    unsigned int p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

void print_xy_slice(Matrix<double> &x, uint3 domain_shape, unsigned int slice_at_z) {
    for (int j = domain_shape.y - 1; j >= 0; j --) {
        for (int i = 0; i < domain_shape.x; i ++) {
            double value = x(IDX(i, j, slice_at_z, domain_shape));
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
    CPML2_Init(&argc, &argv);

    Mapper global(
        uint3{Nx + 2 * Gd, Ny + 2 * Gd, Nz + 2 * Gd},
        uint3{0, 0, 0}
    );
    Matrix<double> ga, gt, gb, gr;
    ga.alloc(global.shape, 7, HDCType::Host  , 0);
    gt.alloc(global.shape, 1, HDCType::Device, 1);
    gb.alloc(global.shape, 1, HDCType::Host  , 2);
    gr.alloc(global.shape, 1, HDCType::Device, 3);
    const double dx = Lx / Nx;
    for (unsigned int i = Gd; i < Gd + Nx; i ++) {
        double ac, ae, aw, bc;
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
        unsigned int idx = IDX(i, Gd, Gd, global.shape);
        ga(idx, 0) = ac;
        ga(idx, 1) = ae;
        ga(idx, 2) = aw;
        gb(idx)    = bc;
    }

    CPM cpm;
    CPML2_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPML2_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = {(unsigned int)cpm.size, 1, 1};
    cpm.init_neighbour();
    printf("%d(%u %u %u): E%2d W%2d N%2d S%2d T%2d B%2d\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, cpm.neighbour[0], cpm.neighbour[1], cpm.neighbour[2], cpm.neighbour[3], cpm.neighbour[4], cpm.neighbour[5]);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(cpm.rank % gpu_count);
    printf("process %d running no device %d\n", cpm.rank, cpm.rank % gpu_count);
    CPML2_Barrier(MPI_COMM_WORLD);

    unsigned int ox = 0, oy = 0, oz = 0;
    for (int i = 0; i < cpm.idx.x; i ++) {
        ox += dim_division(Nx, cpm.shape.x, i);
    }
    for (int j = 0; j < cpm.idx.y; j ++) {
        oy += dim_division(Ny, cpm.shape.y, j);
    }
    for (int k = 0; k < cpm.idx.z; k ++) {
        oz += dim_division(Nz, cpm.shape.z, k);
    }
    Mapper process(
        uint3{dim_division(Nx, cpm.shape.x, cpm.idx.x) + Gdx2, dim_division(Ny, cpm.shape.y, cpm.idx.y) + Gdx2, dim_division(Nz, cpm.shape.z, cpm.idx.z) + Gdx2},
        uint3{ox, oy, oz}
    );
    for (int i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            printf("%-2d(%-2u %-2u %-2u): (%-3u %-3u %-3u) (%-3u %-3u %-3u)\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, process.shape.x, process.shape.y, process.shape.z, process.offset.x, process.offset.y, process.offset.z);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    Matrix<double> a, t, b, r;
    a.alloc(process.shape, 7, HDCType::Host  , 0);
    t.alloc(process.shape, 1, HDCType::Device, 1);
    b.alloc(process.shape, 1, HDCType::Host  , 2);
    r.alloc(process.shape, 1, HDCType::Device, 3);
    for (unsigned int i = Gd; i < process.shape.x - Gd; i ++) {
        for (unsigned int j = Gd; j < process.shape.y - Gd; j ++) {
            for (unsigned int k = Gd; k < process.shape.z - Gd; k ++) {
                unsigned int gi = i + process.offset.x;
                unsigned int gj = j + process.offset.y;
                unsigned int gk = k + process.offset.z;
                unsigned int idx = IDX(i, j, k, process.shape);
                unsigned int gidx = IDX(gi, gj, gk, global.shape);
                for (unsigned int m = 0; m < 7; m ++) {
                    a(idx, m) = ga(gidx, m);
                }
                b(idx) = gb(gidx);
            }
        }
    }

    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, process);
    Matrix<double> region(process.shape, 1, HDCType::Host, -1);
    set_matrix_value(region, inner_shape, inner_offset, process.shape, cpm.rank * 10);
    for (int i = 0; i < 6; i ++) {
        if (cpm.neighbour[i] >= 0) {
            set_matrix_value(region, boundary_shape[i], boundary_offset[i], process.shape, 100 + i);
        }
    }

    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xy_slice(region, process.shape, process.shape.z / 2);
            printf("\n");
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    a.sync(MCpType::Hst2Dev);
    b.sync(MCpType::Hst2Dev);
    dim3 block_dim(32, 1, 1);
    double max_diag = devL2_MaxDiag(a, process, block_dim, cpm);
    printf("%12lf\n", max_diag);
    devL1_ScaleMatrix(a, max_diag, block_dim);
    devL1_ScaleMatrix(b, max_diag, block_dim);
    a.sync(MCpType::Dev2Hst);
    b.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_eq(a, b, process.shape);
            printf("\n");
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    L2EqSolver solver(LSType::PBiCGStab, 10000, 1e-9, 1.2, LSType::SOR, 5, 1.5);
    solver.devL2_Struct3d7p_Solve(a, t, b, r, global, process, block_dim, cpm);
    t.sync(MCpType::Dev2Hst);
    r.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (i == cpm.rank) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_result(t, r, process.shape);
            printf("\n");
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    printf("%d %.12lf\n", solver.it, solver.err);

    return CPML2_Finalize();
}