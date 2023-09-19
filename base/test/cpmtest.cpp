#include <stdio.h>
#include "../src/CPM.h"
#include "../src/CPMDev.h"
#include "../src/matrix.h"

#define Nx   15
#define Ny   12
#define Nz   13
#define Gd   2
#define Gdx2 4

using namespace Falm;

void print_buffer_dev(CPMBuffer<double> &buffer) {
    double *ptr = (double *)falmHostMalloc(sizeof(double) * buffer.size);
    falmMemcpy(ptr, buffer.ptr, sizeof(double) * buffer.size, MCpType::Dev2Hst);
    for (int i = 0; i < buffer.size; i ++) {
        printf("%-2.0lf ", ptr[i]);
    }
    printf("\n");
    falmHostFreePtr(ptr);
}

void print_buffer_host(CPMBuffer<double> &buffer) {
    for (int i = 0; i < buffer.size; i ++) {
        printf("%-2.0lf ", buffer.ptr[i]);
    }
    printf("\n");
}

unsigned int dim_division(unsigned int dim_size, int mpi_size, int mpi_rank) {
    unsigned int inner_dim_size = dim_size - Gdx2;
    unsigned int p_dim_size = inner_dim_size / mpi_size;
    if (mpi_rank < inner_dim_size % mpi_size) {
        p_dim_size ++;
    }
    return p_dim_size;
}

void print_xy_slice(Matrix<double> &x, uint3 domain_shape, unsigned int slice_at_z) {
    for (int j = 0; j < domain_shape.y; j ++) {
        for (int i = 0; i < domain_shape.x; i ++) {
            double value = x(IDX(i, j, slice_at_z, domain_shape));
            if (value == 0) {
                printf(".  ", value);
            } else {
                printf("%-2.0lf ", value);
            }
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    CPM_Init(&argc, &argv);

    Mapper global(
        uint3{Nx + Gdx2, Ny + Gdx2, Nz + Gdx2},
        uint3{0, 0, 0}
    );

    int mpi_size, mpi_rank;
    CPM_GetRank(MPI_COMM_WORLD, mpi_rank);
    CPM_GetSize(MPI_COMM_WORLD, mpi_size);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(mpi_rank % gpu_count);
    printf("process %d running no device %d\n", mpi_rank, mpi_rank % gpu_count);
    CPM_Barrier(MPI_COMM_WORLD);

    unsigned int ox = 0;
    for (int i = 0; i < mpi_rank; i ++) {
        ox += dim_division(global.shape.x, mpi_size, i);
    }
    Mapper process(
        uint3{dim_division(global.shape.x, mpi_size, mpi_rank) + Gdx2, global.shape.y, global.shape.z},
        uint3{ox, 0, 0}
    );

    Matrix<double> x(process.shape, 1, HDCType::Host, 0);
    for (int i = Gd; i < process.shape.x - Gd; i ++) {
        for (int j = Gd; j < process.shape.y - Gd; j ++) {
            for (int k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + mpi_rank * 2;
            }
        }
    }

    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
            printf("\n");
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    CPMBuffer<double> *buffer = new CPMBuffer<double>[8];
    MPI_Request req[8];
    uint3 yz_inner_slice{1, Ny, Nz};
    buffer[0].init(yz_inner_slice, uint3{process.shape.x - Gd - 1, Gd, Gd}, BufType::Out, HDCType::Device, process, Color::Black);
    buffer[1].init(yz_inner_slice, uint3{process.shape.x - Gd    , Gd, Gd}, BufType::In , HDCType::Device, process, Color::Black);
    buffer[2].init(yz_inner_slice, uint3{                  Gd    , Gd, Gd}, BufType::Out, HDCType::Device, process, Color::Black);
    buffer[3].init(yz_inner_slice, uint3{                  Gd - 1, Gd, Gd}, BufType::In , HDCType::Device, process, Color::Black);
    buffer[4].init(yz_inner_slice, uint3{process.shape.x - Gd - 1, Gd, Gd}, BufType::Out, HDCType::Device, process, Color::Red  );
    buffer[5].init(yz_inner_slice, uint3{process.shape.x - Gd    , Gd, Gd}, BufType::In , HDCType::Device, process, Color::Red  );
    buffer[6].init(yz_inner_slice, uint3{                  Gd    , Gd, Gd}, BufType::Out, HDCType::Device, process, Color::Red  );
    buffer[7].init(yz_inner_slice, uint3{                  Gd - 1, Gd, Gd}, BufType::In , HDCType::Device, process, Color::Red  );

    dim3 block_dim_yz(1, 8, 4);

    x.sync(MCpType::Hst2Dev);
    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            dev_CPM_PackColoredBuffer(buffer[0], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[0], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[0]);
            CPM_IRecv(buffer[1], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[1]);
            CPM_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[1], x.dev.ptr, process, block_dim_yz);
        } else if (mpi_rank == mpi_size - 1) {
            dev_CPM_PackColoredBuffer(buffer[2], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[2], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[2]);
            CPM_IRecv(buffer[3], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[3]);
            CPM_Waitall(2, &req[2], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[3], x.dev.ptr, process, block_dim_yz);
        } else {
            dev_CPM_PackColoredBuffer(buffer[0], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[0], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[0]);
            CPM_IRecv(buffer[1], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[1]);
            dev_CPM_PackColoredBuffer(buffer[2], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[2], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[2]);
            CPM_IRecv(buffer[3], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[3]);
            CPM_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[1], x.dev.ptr, process, block_dim_yz);
            dev_CPM_UnpackColoredBuffer(buffer[3], x.dev.ptr, process, block_dim_yz);
        }
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
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
    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            dev_CPM_PackColoredBuffer(buffer[4], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[4], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[4]);
            CPM_IRecv(buffer[5], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[5]);
            CPM_Waitall(2, &req[4], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[5], x.dev.ptr, process, block_dim_yz);
        } else if (mpi_rank == mpi_size - 1) {
            dev_CPM_PackColoredBuffer(buffer[6], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[6], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[6]);
            CPM_IRecv(buffer[7], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[7]);
            CPM_Waitall(2, &req[6], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[7], x.dev.ptr, process, block_dim_yz);
        } else {
            dev_CPM_PackColoredBuffer(buffer[4], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[4], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[4]);
            CPM_IRecv(buffer[5], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[5]);
            dev_CPM_PackColoredBuffer(buffer[6], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[6], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[6]);
            CPM_IRecv(buffer[7], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[7]);
            CPM_Waitall(4, &req[4], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[5], x.dev.ptr, process, block_dim_yz);
            dev_CPM_UnpackColoredBuffer(buffer[7], x.dev.ptr, process, block_dim_yz);
        }
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
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
        CPM_Barrier(MPI_COMM_WORLD);
    }
    // for (int i = 0; i < 8; i ++) {
    //     buffer[i].clear();
    // }
    printf("_____________________________________________________________________\n");
    fflush(stdout);
    CPM_Barrier(MPI_COMM_WORLD);
    
    x.clear(HDCType::HstDev);
    for (int i = Gd; i < process.shape.x - Gd; i ++) {
        for (int j = Gd; j < process.shape.y - Gd; j ++) {
            for (int k = Gd; k < process.shape.z - Gd; k ++) {
                x(IDX(i, j, k, process.shape)) = (i + j + k + SUM3(process.offset)) % 2 + mpi_rank * 2 + 10;
            }
        }
    }
    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
            printf("\n");
        }
        CPM_Barrier(MPI_COMM_WORLD);
    }

    x.sync(MCpType::Hst2Dev);
    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            printf("Sending color %u...\n", Color::Black);
            dev_CPM_PackColoredBuffer(buffer[0], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[0], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[0]);
            CPM_IRecv(buffer[1], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[1]);
            CPM_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[1], x.dev.ptr, process, block_dim_yz);
        } else if (mpi_rank == mpi_size - 1) {
            dev_CPM_PackColoredBuffer(buffer[2], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[2], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[2]);
            CPM_IRecv(buffer[3], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[3]);
            CPM_Waitall(2, &req[2], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[3], x.dev.ptr, process, block_dim_yz);
        } else {
            dev_CPM_PackColoredBuffer(buffer[0], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[0], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[0]);
            CPM_IRecv(buffer[1], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[1]);
            dev_CPM_PackColoredBuffer(buffer[2], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[2], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[2]);
            CPM_IRecv(buffer[3], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[3]);
            CPM_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[1], x.dev.ptr, process, block_dim_yz);
            dev_CPM_UnpackColoredBuffer(buffer[3], x.dev.ptr, process, block_dim_yz);
        }
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
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
    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            printf("Sending color %u...\n", Color::Red);
            dev_CPM_PackColoredBuffer(buffer[4], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[4], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[4]);
            CPM_IRecv(buffer[5], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[5]);
            CPM_Waitall(2, &req[4], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[5], x.dev.ptr, process, block_dim_yz);
        } else if (mpi_rank == mpi_size - 1) {
            dev_CPM_PackColoredBuffer(buffer[6], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[6], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[6]);
            CPM_IRecv(buffer[7], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[7]);
            CPM_Waitall(2, &req[6], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[7], x.dev.ptr, process, block_dim_yz);
        } else {
            dev_CPM_PackColoredBuffer(buffer[4], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[4], mpi_rank + 1, 0, MPI_COMM_WORLD, &req[4]);
            CPM_IRecv(buffer[5], mpi_rank + 1, 1, MPI_COMM_WORLD, &req[5]);
            dev_CPM_PackColoredBuffer(buffer[6], x.dev.ptr, process, block_dim_yz);
            CPM_ISend(buffer[6], mpi_rank - 1, 1, MPI_COMM_WORLD, &req[6]);
            CPM_IRecv(buffer[7], mpi_rank - 1, 0, MPI_COMM_WORLD, &req[7]);
            CPM_Waitall(4, &req[4], MPI_STATUSES_IGNORE);
            dev_CPM_UnpackColoredBuffer(buffer[5], x.dev.ptr, process, block_dim_yz);
            dev_CPM_UnpackColoredBuffer(buffer[7], x.dev.ptr, process, block_dim_yz);
        }
    }
    x.sync(MCpType::Dev2Hst);
    for (int i = 0; i < mpi_size; i ++) {
        if (mpi_rank == i) {
            printf("%d printing...\n", mpi_rank);
            print_xy_slice(x, process.shape, process.shape.z / 2);
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
        CPM_Barrier(MPI_COMM_WORLD);
    }

    CPM_Finalize();

    delete[] buffer;
    return 0;
}