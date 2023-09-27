#include <stdio.h>
#include <typeinfo>
#include "../src/CPML2v2.h"
#include "../src/matrix.h"

#define Nx   25
#define Ny   12
#define Nz   13

#define USE_CUDA_AWARE_MPI true
#define THICK 2

using namespace Falm;

void print_xy_slice(Matrix<REAL> &x, INTx3 domain_shape, INTx3 domain_offset) {
    for (INT j = domain_shape.y - 1; j >= 0; j --) {
        printf(" ");
        for (INT i = 0; i < domain_shape.x; i ++) {
            REAL value = x(IDX(i, j, (Nz + Gdx2) / 2 - domain_offset.z, domain_shape));
            if (value == 0) {
                printf(".   ", value);
            } else {
                printf("%-3.0lf ", value);
            }
        }
        printf("\n");
    }
    printf("yx\n");
}

void print_xz_slice(Matrix<REAL> &x, INTx3 domain_shape, INTx3 domain_offset) {
    for (INT k = domain_shape.z - 1; k >= 0; k --) {
        printf(" ");
        for (INT i = 0; i < domain_shape.x; i ++) {
            REAL value = x(IDX(i, (Ny + Gdx2) / 2 - domain_offset.y, k, domain_shape));
            if (value == 0) {
                printf(".   ", value);
            } else {
                printf("%-3.0lf ", value);
            }
        }
        printf("\n");
    }
    printf("zx\n");
}

INT dim_division(INT dim_size, INT mpi_size, INT mpi_rank) {
    INT p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

int main(int argc, char **argv) {
    CPML2_Init(&argc, &argv);

    Mapper global(
        INTx3{Nx + Gdx2, Ny + Gdx2, Nz + Gdx2},
        INTx3{0, 0, 0}
    );

    CPMBase cpm;
    CPML2_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPML2_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = INTx3{atoi(argv[1]), atoi(argv[2]), atoi(argv[3])};
    if (PRODUCT3(cpm.shape) != cpm.size) {
        printf("wrong group shape: %ux%ux%u != %d\n",cpm.shape.x, cpm.shape.y, cpm.shape.z, cpm.size);
        CPML2_Finalize();
        return 0;
    }
    printf("group shape %ux%ux%u\n", cpm.shape.x, cpm.shape.y, cpm.shape.z);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);
    cpm.initNeighbour();
    printf("%d(%u %u %u): E%2d W%2d N%2d S%2d T%2d B%2d\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, cpm.neighbour[0], cpm.neighbour[1], cpm.neighbour[2], cpm.neighbour[3], cpm.neighbour[4], cpm.neighbour[5]);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(cpm.rank % gpu_count);
    printf("process %d running no device %d\n", cpm.rank, cpm.rank % gpu_count);
    CPML2_Barrier(MPI_COMM_WORLD);

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
    Mapper process(
        INTx3{dim_division(Nx, cpm.shape.x, cpm.idx.x) + Gdx2, dim_division(Ny, cpm.shape.y, cpm.idx.y) + Gdx2, dim_division(Nz, cpm.shape.z, cpm.idx.z) + Gdx2},
        INTx3{ox, oy, oz}
    );
    printf("%d(%u %u %u): (%u %u %u) (%u %u %u))\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, process.shape.x, process.shape.y, process.shape.z, process.offset.x, process.offset.y, process.offset.z);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);

    Matrix<REAL> x(process.shape, 1, HDCType::Host, 0);
    for (INT i = Gd; i < process.shape.x - Gd; i ++) {
        for (INT j = Gd; j < process.shape.y - Gd; j ++) {
            for (INT k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 10;
            }
        }
    }
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    cpm.use_cuda_aware_mpi = USE_CUDA_AWARE_MPI;
    CPMOp<REAL> cpmop(cpm);
    printf("cpmop %d: %d %u\n", cpm.rank, cpmop.mpi_dtype == MPI_DOUBLE, cpmop.buffer_hdctype);
    CPML2_Barrier(MPI_COMM_WORLD);
    INT thick = THICK;

    x.sync(MCpType::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        // printf("%p\n", x.dev.ptr);
        CPML2_Barrier(MPI_COMM_WORLD);
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Black, thick, 0);
        // cpmop.CPML2dev_IExchange6Face(x.dev.ptr, process, thick, 0);
        cpmop.CPML2_Wait6Face();
        // printf("%p\n", x.dev.ptr);
        printf("%d sending complete\n", cpm.rank);
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        // cpmop.CPML2dev_PostExchange6Face();
    }
    x.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            // printf("B0: ");
            // print_buffer_dev(buffer[0]);
            // printf("B1: ");
            // print_buffer_dev(buffer[1]);
            // printf("B2: ");
            // print_buffer_dev(buffer[2]);
            // printf("B3: ");
            // print_buffer_dev(buffer[3]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Red, thick, 0);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
    }
    x.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            // printf("B0: ");
            // print_buffer_dev(buffer[0]);
            // printf("B1: ");
            // print_buffer_dev(buffer[1]);
            // printf("B2: ");
            // print_buffer_dev(buffer[2]);
            // printf("B3: ");
            // print_buffer_dev(buffer[3]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    x.release(HDCType::HstDev);
    
    printf("_____________________________________________________________________\n");
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);

    x.alloc(process.shape, 1, HDCType::Host, 0);
    for (INT i = Gd; i < process.shape.x - Gd; i ++) {
        for (INT j = Gd; j < process.shape.y - Gd; j ++) {
            for (INT k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 100;
            }
        }
    }
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    x.sync(MCpType::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        // printf("%u %p\n", x.hdctype, x.dev.ptr);
        CPML2_Barrier(MPI_COMM_WORLD);
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Black, thick, 0);
        // cpmop.CPML2dev_IExchange6Face(x.dev.ptr, process, thick, 0);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        // cpmop.CPML2dev_PostExchange6Face();
    }
    x.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            // printf("B0: ");
            // print_buffer_dev(buffer[0]);
            // printf("B1: ");
            // print_buffer_dev(buffer[1]);
            // printf("B2: ");
            // print_buffer_dev(buffer[2]);
            // printf("B3: ");
            // print_buffer_dev(buffer[3]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Red, thick, 0);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
    }
    x.sync(MCpType::Dev2Hst);
    for (INT i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            if (cpm.shape.y == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            // printf("B0: ");
            // print_buffer_dev(buffer[0]);
            // printf("B1: ");
            // print_buffer_dev(buffer[1]);
            // printf("B2: ");
            // print_buffer_dev(buffer[2]);
            // printf("B3: ");
            // print_buffer_dev(buffer[3]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    x.release(HDCType::HstDev);

    return CPML2_Finalize();
}
