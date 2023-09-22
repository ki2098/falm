#include <stdio.h>
#include <stdlib.h>
#include "../src/CPML2.h"
#include "../src/matrix.h"

#define Nx   15
#define Ny   12
#define Nz   9


using namespace Falm;

void print_buffer_dev(CPMBuffer<double> &buffer) {
    double *ptr = (double *)falmHostMalloc(sizeof(double) * buffer.size);
    falmMemcpy(ptr, buffer.ptr, sizeof(double) * buffer.size, MCpType::Dev2Hst);
    for (int i = 0; i < buffer.size; i ++) {
        printf("%-3.0lf ", ptr[i]);
    }
    printf("\n");
    falmHostFreePtr(ptr);
}

void print_buffer_host(CPMBuffer<double> &buffer) {
    for (int i = 0; i < buffer.size; i ++) {
        printf("%-3.0lf ", buffer.ptr[i]);
    }
    printf("\n");
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

void print_xz_slice(Matrix<double> &x, uint3 domain_shape, unsigned int slice_at_y) {
    for (int k = domain_shape.z - 1; k >= 0; k --) {
        for (int i = 0; i < domain_shape.x; i ++) {
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

int main(int argc, char **argv) {
    CPML2_Init(&argc, &argv);

    Mapper global(
        uint3{Nx + Gdx2, Ny + Gdx2, Nz + Gdx2},
        uint3{0, 0, 0}
    );

    CPM cpm;
    CPML2_GetRank(MPI_COMM_WORLD, cpm.rank);
    CPML2_GetSize(MPI_COMM_WORLD, cpm.size);
    cpm.shape = uint3{(unsigned int)atoi(argv[1]), (unsigned int)atoi(argv[2]), (unsigned int)atoi(argv[3])};
    if (PRODUCT3(cpm.shape) != cpm.size) {
        printf("wrong group shape: %ux%ux%u != %d\n",cpm.shape.x, cpm.shape.y, cpm.shape.z, cpm.size);
        CPML2_Finalize();
        return 0;
    }
    printf("group shape %ux%ux%u\n", cpm.shape.x, cpm.shape.y, cpm.shape.z);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);
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
    printf("%d(%u %u %u): (%u %u %u) (%u %u %u))\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z, process.shape.x, process.shape.y, process.shape.z, process.offset.x, process.offset.y, process.offset.z);
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);

    Matrix<double> x(process.shape, 1, HDCType::Host, 0);
    for (int i = Gd; i < process.shape.x - Gd; i ++) {
        for (int j = Gd; j < process.shape.y - Gd; j ++) {
            for (int k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 10;
            }
        }
    }

    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
            printf("\n");
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    MPI_Request *req;
    CPMBuffer<double> *buffer;
    unsigned int bufhdc = HDCType::Host;
    
    dim3 block_dim_yz(1, 8, 4);

    x.sync(MCpType::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Black, 2, 0, buffer, bufhdc, req);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, process, Color::Black, buffer, req);
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
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
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Red, 2, 0, buffer, bufhdc, req);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, process, Color::Red, buffer, req);
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
            printf("\n");
            // printf("B4: ");
            // print_buffer_dev(buffer[4]);
            // printf("B5: ");
            // print_buffer_dev(buffer[5]);
            // printf("B6: ");
            // print_buffer_dev(buffer[6]);
            // printf("B7: ");
            // print_buffer_dev(buffer[7]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    // for (int i = 0; i < 8; i ++) {
    //     buffer[i].clear();
    // }
    x.release(HDCType::HstDev);
    
    printf("_____________________________________________________________________\n");
    fflush(stdout);
    CPML2_Barrier(MPI_COMM_WORLD);
    
    x.alloc(process.shape, 1, HDCType::Host, 0);
    for (int i = Gd; i < process.shape.x - Gd; i ++) {
        for (int j = Gd; j < process.shape.y - Gd; j ++) {
            for (int k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 100;
            }
        }
    }
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
            printf("\n");
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }

    x.sync(MCpType::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Black, 2, 0, buffer, bufhdc, req);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, process, Color::Black, buffer, req);
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
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
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, process, Color::Red, 2, 0, buffer, bufhdc, req);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, process, Color::Red, buffer, req);
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx.x, cpm.idx.y, cpm.idx.z);
            print_xz_slice(x, process.shape, process.shape.y / 2);
            printf("\n");
            // printf("B4: ");
            // print_buffer_dev(buffer[4]);
            // printf("B5: ");
            // print_buffer_dev(buffer[5]);
            // printf("B6: ");
            // print_buffer_dev(buffer[6]);
            // printf("B7: ");
            // print_buffer_dev(buffer[7]);
            fflush(stdout);
        }
        CPML2_Barrier(MPI_COMM_WORLD);
    }
    x.release(HDCType::HstDev);
    

    CPML2_Finalize();
    
    return 0;
}
