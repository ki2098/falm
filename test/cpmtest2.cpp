#include <stdio.h>
#include <stdlib.h>
#include "../src/CPML2v2.h"
#include "../src/matrix.h"

#define Nx   15
#define Ny   12
#define Nz   9

using namespace Falm;

Int dim_division(Int dim_size, Int mpi_size, Int mpi_rank) {
    Int p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

void print_xy_slice(Matrix<double> &x, Int3 domain_shape, Int slice_at_z) {
    for (Int j = domain_shape[1] - 1; j >= 0; j --) {
        for (Int i = 0; i < domain_shape[0]; i ++) {
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

void print_xz_slice(Matrix<double> &x, Int3 domain_shape, Int slice_at_y) {
    for (Int k = domain_shape[2] - 1; k >= 0; k --) {
        for (Int i = 0; i < domain_shape[0]; i ++) {
            double value = x(IDX(i, slice_at_y, k, domain_shape));
            if (value == 0) {
                printf(".   ", value);
            } else {
                printf("%-3.0lf ", value);
            }
        }
        printf("\n");
    }
}

void set_matrix_value(Matrix<double> &x, Int3 range_shape, Int3 range_offset, Int3 pshape, double value) {
    for (Int i = 0; i < range_shape[0]; i ++) {
        for (Int j = 0; j < range_shape[1]; j ++) {
            for (Int k = 0; k < range_shape[2]; k ++) {
                Int _i = i + range_offset[0];
                Int _j = j + range_offset[1];
                Int _k = k + range_offset[2];
                x(IDX(_i, _j, _k, pshape)) = value;
            }
        }
    }
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);

    Region global(
        Int3{Nx + Gdx2, Ny + Gdx2, Nz + Gdx2},
        Int3{0, 0, 0}
    );

    CPMBase cpm;
    CPM_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPM_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = Int3{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    if (PRODUCT3(cpm.shape) != cpm.size) {
        printf("wrong group shape: %ux%ux%u != %d\n",cpm.shape[0], cpm.shape[1], cpm.shape[2], cpm.size);
        CPM_Finalize();
        return 0;
    }
    printf("group shape %ux%ux%u\n", cpm.shape[0], cpm.shape[1], cpm.shape[2]);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);
    cpm.initNeighbour();
    printf("%d(%u %u %u): E%2d W%2d N%2d S%2d T%2d B%2d\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2], cpm.neighbour[0], cpm.neighbour[1], cpm.neighbour[2], cpm.neighbour[3], cpm.neighbour[4], cpm.neighbour[5]);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    Int ox = 0, oy = 0, oz = 0;
    for (Int i = 0; i < cpm.idx[0]; i ++) {
        ox += dim_division(Nx, cpm.shape[0], i);
    }
    for (Int j = 0; j < cpm.idx[1]; j ++) {
        oy += dim_division(Ny, cpm.shape[1], j);
    }
    for (Int k = 0; k < cpm.idx[2]; k ++) {
        oz += dim_division(Nz, cpm.shape[2], k);
    }
    Region process(
        Int3{dim_division(Nx, cpm.shape[0], cpm.idx[0]) + Gdx2, dim_division(Ny, cpm.shape[1], cpm.idx[1]) + Gdx2, dim_division(Nz, cpm.shape[2], cpm.idx[2]) + Gdx2},
        Int3{ox, oy, oz}
    );
    printf("%d(%u %u %u): (%u %u %u) (%u %u %u))\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2], process.shape[0], process.shape[1], process.shape[2], process.offset[0], process.offset[1], process.offset[2]);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    Int3 inner_shape, inner_offset;
    Int3 boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(process, Gd));
    Matrix<double> x(process.shape, 1, HDC::Host);
    set_matrix_value(x, inner_shape, inner_offset, process.shape, cpm.rank * 10);
    for (Int i = 0; i < 6; i ++) {
        if (cpm.neighbour[i] >= 0) {
            set_matrix_value(x, boundary_shape[i], boundary_offset[i], process.shape, 100 + i);
        }
    }

    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            print_xy_slice(x, process.shape, process.shape[2] / 2);
            printf("\n");
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    CPM_Finalize();

    return 0;
}