#include <stdio.h>
#include <typeinfo>
#include <cuda_profiler_api.h>
#include "../src/CPML2v2.h"
#include "../src/matrix.h"

#define Nx   25
#define Ny   22
#define Nz   24

#define USE_CUDA_AWARE_MPI true
#define THICK 1
#define MARGIN 1

Falm::Stream faceStream[6];
#define FACESTREAM faceStream

using namespace Falm;

void print_xy_slice(Matrix<Real> &x, Int3 domain_shape, Int3 domain_offset) {
    for (Int j = domain_shape[1] - 1; j >= 0; j --) {
        printf(" ");
        for (Int i = 0; i < domain_shape[0]; i ++) {
            Real value = x(IDX(i, j, (Nz + Gdx2) / 2 - domain_offset[2], domain_shape));
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

void print_xz_slice(Matrix<Real> &x, Int3 domain_shape, Int3 domain_offset) {
    for (Int k = domain_shape[2] - 1; k >= 0; k --) {
        printf(" ");
        for (Int i = 0; i < domain_shape[0]; i ++) {
            Real value = x(IDX(i, (Ny + Gdx2) / 2 - domain_offset[1], k, domain_shape));
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

Int dim_division(Int dim_size, Int mpi_size, Int mpi_rank) {
    Int p_dim_size = dim_size / mpi_size;
    if (mpi_rank < dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
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

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(cpm.rank % gpu_count);
    printf("process %d running no device %d\n", cpm.rank, cpm.rank % gpu_count);
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

    Matrix<Real> x(process.shape, 1, HDC::Host, "x");
    for (Int i = Gd; i < process.shape[0] - Gd; i ++) {
        for (Int j = Gd; j < process.shape[1] - Gd; j ++) {
            for (Int k = Gd; k < process.shape[2] - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 10;
            }
        }
    }
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    cpm.use_cuda_aware_mpi = USE_CUDA_AWARE_MPI;
    CPMComm<Real> cpmop(cpm);
    printf("cpmop %d: %d %u\n", cpm.rank, cpmop.mpi_dtype == MPI_DOUBLE, cpmop.buffer_hdctype);
    CPM_Barrier(MPI_COMM_WORLD);
    Int thick = THICK;
    Int margin = MARGIN;
    
    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamCreate(&faceStream[fid]);
    }
    
    x.sync(MCP::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        // printf("%p\n", x.dev.ptr);
        CPM_Barrier(MPI_COMM_WORLD);
        cpmop.IExchange6ColoredFace(x.dev.ptr, process, Color::Black, thick, margin, 0, FACESTREAM);
        // cpmop.CPML2dev_IExchange6Face(x.dev.ptr, process, thick, 0);
        cpmop.CPML2_Wait6Face();
        // printf("%p\n", x.dev.ptr);
        printf("%d sending complete\n", cpm.rank);
        cpmop.PostExchange6ColoredFace(FACESTREAM);
        // cpmop.CPML2dev_PostExchange6Face();
    }
    x.sync(MCP::Dev2Hst);
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
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
        CPM_Barrier(MPI_COMM_WORLD);
    }
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
        cpmop.IExchange6ColoredFace(x.dev.ptr, process, Color::Red, thick, margin, 0, FACESTREAM);
        cpmop.CPML2_Wait6Face();
        cpmop.PostExchange6ColoredFace(FACESTREAM);
    }
    x.sync(MCP::Dev2Hst);
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
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
        CPM_Barrier(MPI_COMM_WORLD);
    }

    x.release(HDC::HstDev);
    
    printf("_____________________________________________________________________\n");
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);

    x.alloc(process.shape, 1, HDC::Host);
    for (Int i = Gd; i < process.shape[0] - Gd; i ++) {
        for (Int j = Gd; j < process.shape[1] - Gd; j ++) {
            for (Int k = Gd; k < process.shape[2] - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + cpm.rank * 100;
            }
        }
    }
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
                print_xz_slice(x, process.shape, process.offset);
            else
                print_xy_slice(x, process.shape, process.offset);
            printf("\n");
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    x.sync(MCP::Hst2Dev);
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            fflush(stdout);
        }
        // printf("%u %p\n", x.hdctype, x.dev.ptr);
        CPM_Barrier(MPI_COMM_WORLD);
        cpmop.IExchange6ColoredFace(x.dev.ptr, process, Color::Black, thick, margin, 0, FACESTREAM);
        // cpmop.CPML2dev_IExchange6Face(x.dev.ptr, process, thick, 0);
        cpmop.CPML2_Wait6Face();
        cpmop.PostExchange6ColoredFace(FACESTREAM);
        // cpmop.CPML2dev_PostExchange6Face();
    }
    x.sync(MCP::Dev2Hst);
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
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
        CPM_Barrier(MPI_COMM_WORLD);
    }
    if (cpm.size > 1) {
        if (cpm.rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            fflush(stdout);
        }
        CPM_Barrier(MPI_COMM_WORLD);
        cpmop.IExchange6ColoredFace(x.dev.ptr, process, Color::Red, thick, margin, 0, FACESTREAM);
        cpmop.CPML2_Wait6Face();
        cpmop.PostExchange6ColoredFace(FACESTREAM);
    }
    x.sync(MCP::Dev2Hst);
    for (Int i = 0; i < cpm.size; i ++) {
        if (cpm.rank == i) {
            printf("%d(%u %u %u) printing...\n", cpm.rank, cpm.idx[0], cpm.idx[1], cpm.idx[2]);
            if (cpm.shape[1] == 1) 
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
        CPM_Barrier(MPI_COMM_WORLD);
    }

    x.release(HDC::HstDev);

    for (int fid = 0; fid < 6; fid ++) {
        cudaStreamDestroy(faceStream[fid]);
    }

    return CPM_Finalize();
}
