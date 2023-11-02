#include <stdio.h>
#include <stdlib.h>
#include "../src/CPML2v2.h"
#include "../src/matrix.h"

#define Nx   15
#define Ny   12
#define Nz   9

using namespace Falm;

INT dim_division(INT dim_size, INT mpi_size, INT mpi_rank) {
    INT p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

void print_xy_slice(Matrix<double> &x, INT3 domain_shape, INT slice_at_z) {
    for (INT j = domain_shape.y - 1; j >= 0; j --) {
        for (INT i = 0; i < domain_shape.x; i ++) {
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

void print_xz_slice(Matrix<double> &x, INT3 domain_shape, INT slice_at_y) {
    for (INT k = domain_shape.z - 1; k >= 0; k --) {
        for (INT i = 0; i < domain_shape.x; i ++) {
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

void set_matrix_value(Matrix<double> &x, INT3 range_shape, INT3 range_offset, INT3 pshape, double value) {
    for (INT i = 0; i < range_shape.x; i ++) {
        for (INT j = 0; j < range_shape.y; j ++) {
            for (INT k = 0; k < range_shape.z; k ++) {
                INT _i = i + range_offset.x;
                INT _j = j + range_offset.y;
                INT _k = k + range_offset.z;
                x(IDX(_i, _j, _k, pshape)) = value;
            }
        }
    }
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);

    Region global(
        INT3{Nx + Gdx2, Ny + Gdx2, Nz + Gdx2},
        INT3{0, 0, 0}
    );

    CPMBase cpm;
    CPM_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPM_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = INT3{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    if (PRODUCT3(cpm.shape) != cpm.size) {
        printf("wrong group shape: %ux%ux%u != %d\n",cpm.shape.x, cpm.shape.y, cpm.shape.z, cpm.size);
        CPM_Finalize();
        return 0;
    }
    printf("group shape %ux%ux%u\n", cpm.shape.x, cpm.shape.y, cpm.shape.z);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);
    cpm.initNeighbour();
    printf("%d(%u %u %u): E%2d W%2d N%2d S%2d T%2d B%2d\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, cpm.neighbour[0], cpm.neighbour[1], cpm.neighbour[2], cpm.neighbour[3], cpm.neighbour[4], cpm.neighbour[5]);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    INT ox = 0, oy = 0, oz = 0;
    for (INT i = 0; i < cpm.idx.x; i ++) {
        ox += dim_division(Nx, cpm.shape.x, i);
    }
    for (INT j = 0; j < cpm.idx.y; j ++) {
        oy += dim_division(Ny, cpm.shape.y, j);
    }
    for (INT k = 0; k < cpm.idx.z; k ++) {
        oz += dim_division(Nz, cpm.shape.z, k);
    }
    Region process(
        INT3{dim_division(Nx, cpm.shape.x, cpm.idx.x) + Gdx2, dim_division(Ny, cpm.shape.y, cpm.idx.y) + Gdx2, dim_division(Nz, cpm.shape.z, cpm.idx.z) + Gdx2},
        INT3{ox, oy, oz}
    );
    printf("%d(%u %u %u): (%u %u %u) (%u %u %u))\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, process.shape.x, process.shape.y, process.shape.z, process.offset.x, process.offset.y, process.offset.z);
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    INT3 inner_shape, inner_offset;
    INT3 boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(process, Gd));
    Matrix<double> x(process.shape, 1, HDCType::Host);
    set_matrix_value(x, inner_shape, inner_offset, process.shape, cpm.rank * 10);
    for (INT i = 0; i < 6; i ++) {
        if (cpm.neighbour[i] >= 0) {
            set_matrix_value(x, boundary_shape[i], boundary_offset[i], process.shape, 100 + i);
        }
    }

    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xy_slice(x, process.shape, process.shape.z / 2);
            printf("\n");
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    CPM_Finalize();

    return 0;
}