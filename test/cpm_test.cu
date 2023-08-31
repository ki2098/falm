#include <cstdio>
#include "../base/src/Field.cuh"
#include "../base/src/Mapper.cuh"
#include "../base/src/CPM.cuh"
#include "../base/src/Util.cuh"

using namespace FALM;


const int XY_PLAIN = 0;
const int XZ_PLAIN = 1;
const int YZ_PLAIN = 2;

const unsigned int __g = 2 * guide - 1;

unsigned int cells[3] = {10, 14, 11};

unsigned int calc_inner(unsigned int global, int mpi_size, int mpi_rank) {
    unsigned int global_inner = global - 2 * guide;
    unsigned int local_inner = global_inner / mpi_size;
    if (mpi_rank < global_inner % mpi_size) {
        local_inner ++;
    }
    return local_inner;
}

void print_slice(Field<double> &x, Mapper &domain, int slice_plain, int slice_at) {
    dim3 &size = domain.size;
    if (slice_plain == XY_PLAIN) {
        for (int j = 0; j < size.y; j ++) {
            for (int i = 0; i < size.x; i ++) {
                unsigned int idx = UTIL::IDX(i, j, slice_at, size);
                if (x(idx) == 0.) {
                    printf("  . ");
                } else {
                    printf("%3.lf ", x(idx));
                }
            }
            printf("\n");
        }
    } else if (slice_plain == XZ_PLAIN) {
        for (int k = 0; k < size.z; k ++) {
            for (int i = 0; i < size.x; i ++) {
                unsigned int idx = UTIL::IDX(i, slice_at, k, size);
                if (x(idx) == 0.) {
                    printf("  . ");
                } else {
                    printf("%3.lf ", x(idx));
                }
            }
            printf("\n");
        }
    } else if (slice_plain == YZ_PLAIN) {
        for (int k = 0; k < size.z; k ++) {
            for (int j = 0; j < size.y; j ++) {
                unsigned int idx = UTIL::IDX(slice_at, j, k, size);
                if (x(idx) == 0.) {
                    printf("  . ");
                } else {
                    printf("%3.lf ", x(idx));
                }
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv) {
    MPI_State mpi;

    CUDA_ERROR_DEINITIALIZED;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);

    Mapper global(
        dim3(cells[0] + __g, cells[1] + __g, cells[2] + __g),
        dim3(0, 0, 0)
    );
    if (mpi.rank == 0) {
        printf("global=(%u %u %u)\n", global.size.x, global.size.y, global.size.z);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int local_inner_x = calc_inner(global.size.x, mpi.size, mpi.rank);
    unsigned int local_offset_x = 0;
    for (int i = 0; i < mpi.rank; i ++) {
        local_offset_x += calc_inner(global.size.x, mpi.size, i);
    }
    Mapper domain(
        dim3(local_inner_x + 2 * guide, global.size.y, global.size.z),
        dim3(local_offset_x, 0, 0)
    );

    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("rank=%d size=(%u %u %u) offset=(%u %u %u)\n", mpi.rank, domain.size.x, domain.size.y, domain.size.z, domain.offset.x, domain.offset.y, domain.offset.z);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    unsigned int slice_at = guide + 1;

    Field<double> x(domain.size, 1, LOC::HOST, 0);
    for (int i = guide; i < domain.size.x - guide; i ++) {
        for (int j = guide; j < domain.size.y - guide; j ++) {
            for (int k = guide; k < domain.size.z - guide; k ++) {
                unsigned int idx = UTIL::IDX(i, j, k, domain.size);
                if ((i + j + k + domain.offset.x + domain.offset.y + domain.offset.z) % 2 == 0) {
                    x(idx) = mpi.rank*10+10;
                } else {
                    x(idx) = mpi.rank*10+11;
                }
            }
        }
    }
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    x.sync(SYNC::H2D);

    CPM::Request req[4];
    MPI_Request mpi_req[4];
    for (int i = 0; i < 4; i ++) {
        req[i].request = &(mpi_req[i]);
    }

    dim3 &size = domain.size;
    const unsigned int g = guide;
    dim3 yz_inner_slice(1, size.y - 2 * g, size.z - 2 * g);

    if (mpi.rank == 0) {
        req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
        req[1].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(yz_inner_slice, dim3(       g  , g, g));
        req[1].map.set(yz_inner_slice, dim3(       g-1, g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
        req[1].map.set(yz_inner_slice, dim3(       g  , g, g));
        req[2].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
        req[3].map.set(yz_inner_slice, dim3(       g-1, g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 0...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 0, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 0, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (mpi.rank == 0) {
        req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
        req[1].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(yz_inner_slice, dim3(       g  , g, g));
        req[1].map.set(yz_inner_slice, dim3(       g-1, g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
        req[1].map.set(yz_inner_slice, dim3(       g  , g, g));
        req[2].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
        req[3].map.set(yz_inner_slice, dim3(       g-1, g, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 1...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 1, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 1, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

/*-----------------------------------------------------------------------------------------------------------*/
    x.release(LOC::BOTH);
    printf("________________________________________\n");
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int local_inner_y = calc_inner(global.size.y, mpi.size, mpi.rank);
    unsigned int local_offset_y = 0;
    for (int i = 0; i < mpi.rank; i ++) {
        local_offset_y += calc_inner(global.size.y, mpi.size, i);
    }
    domain.set(
        dim3(global.size.x, local_inner_y + 2 * guide, global.size.z),
        dim3(0, local_offset_y, 0)
    );
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("rank=%d size=(%u %u %u) offset=(%u %u %u)\n", mpi.rank, domain.size.x, domain.size.y, domain.size.z, domain.offset.x, domain.offset.y, domain.offset.z);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    x.init(domain.size, 1, LOC::HOST, 0);
    for (int i = guide; i < domain.size.x - guide; i ++) {
        for (int j = guide; j < domain.size.y - guide; j ++) {
            for (int k = guide; k < domain.size.z - guide; k ++) {
                unsigned int idx = UTIL::IDX(i, j, k, domain.size);
                if ((i + j + k + domain.offset.x + domain.offset.y + domain.offset.z) % 2 == 0) {
                    x(idx) = mpi.rank*10+10;
                } else {
                    x(idx) = mpi.rank*10+11;
                }
            }
        }
    }
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    x.sync(SYNC::H2D);
    

    dim3 xz_inner_slice(size.x - 2 * g, 1, size.z - 2 * g);

    if (mpi.rank == 0) {
        req[0].map.set(xz_inner_slice, dim3(g, size.y-g-1, g));
        req[1].map.set(xz_inner_slice, dim3(g, size.y-g  , g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(xz_inner_slice, dim3(g,        g  , g));
        req[1].map.set(xz_inner_slice, dim3(g,        g-1, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(xz_inner_slice, dim3(g, size.y-g-1, g));
        req[1].map.set(xz_inner_slice, dim3(g,        g  , g));
        req[2].map.set(xz_inner_slice, dim3(g, size.y-g  , g));
        req[3].map.set(xz_inner_slice, dim3(g,        g-1, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 0...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 0, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 0, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (mpi.rank == 0) {
        req[0].map.set(xz_inner_slice, dim3(g, size.y-g-1, g));
        req[1].map.set(xz_inner_slice, dim3(g, size.y-g  , g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(xz_inner_slice, dim3(g,        g  , g));
        req[1].map.set(xz_inner_slice, dim3(g,        g-1, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(xz_inner_slice, dim3(g, size.y-g-1, g));
        req[1].map.set(xz_inner_slice, dim3(g,        g  , g));
        req[2].map.set(xz_inner_slice, dim3(g, size.y-g  , g));
        req[3].map.set(xz_inner_slice, dim3(g,        g-1, g));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 1...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 1, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 1, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XY_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

/*-----------------------------------------------------------------------------------------------------------*/
    x.release(LOC::BOTH);
    printf("________________________________________\n");
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    unsigned int local_inner_z = calc_inner(global.size.z, mpi.size, mpi.rank);
    unsigned int local_offset_z = 0;
    for (int i = 0; i < mpi.rank; i ++) {
        local_offset_z += calc_inner(global.size.z, mpi.size, i);
    }
    domain.set(
        dim3(global.size.x, global.size.y, local_inner_z + 2 * guide),
        dim3(0, 0, local_offset_z)
    );
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("rank=%d size=(%u %u %u) offset=(%u %u %u)\n", mpi.rank, domain.size.x, domain.size.y, domain.size.z, domain.offset.x, domain.offset.y, domain.offset.z);
            fflush(stdout);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    x.init(domain.size, 1, LOC::HOST, 0);
    for (int i = guide; i < domain.size.x - guide; i ++) {
        for (int j = guide; j < domain.size.y - guide; j ++) {
            for (int k = guide; k < domain.size.z - guide; k ++) {
                unsigned int idx = UTIL::IDX(i, j, k, domain.size);
                if ((i + j + k + domain.offset.x + domain.offset.y + domain.offset.z) % 2 == 0) {
                    x(idx) = mpi.rank*10+10;
                } else {
                    x(idx) = mpi.rank*10+11;
                }
            }
        }
    }
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XZ_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    x.sync(SYNC::H2D);

    dim3 xy_inner_slice(size.x - 2 * g, size.y - 2 * g, 1);

    if (mpi.rank == 0) {
        req[0].map.set(xy_inner_slice, dim3(g, g, size.z-g-1));
        req[1].map.set(xy_inner_slice, dim3(g, g, size.z-g  ));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(xy_inner_slice, dim3(g, g,        g  ));
        req[1].map.set(xy_inner_slice, dim3(g, g,        g-1));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(xy_inner_slice, dim3(g, g, size.z-g-1));
        req[1].map.set(xy_inner_slice, dim3(g, g,        g  ));
        req[2].map.set(xy_inner_slice, dim3(g, g, size.z-g  ));
        req[3].map.set(xy_inner_slice, dim3(g, g,        g-1));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 0, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 0, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 0, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 0...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 0, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 0, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 0, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XZ_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (mpi.rank == 0) {
        req[0].map.set(xy_inner_slice, dim3(g, g, size.z-g-1));
        req[1].map.set(xy_inner_slice, dim3(g, g, size.z-g  ));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
    } else if (mpi.rank == mpi.size - 1) {
        req[0].map.set(xy_inner_slice, dim3(g, g,        g  ));
        req[1].map.set(xy_inner_slice, dim3(g, g,        g-1));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    } else {
        req[0].map.set(xy_inner_slice, dim3(g, g, size.z-g-1));
        req[1].map.set(xy_inner_slice, dim3(g, g,        g  ));
        req[2].map.set(xy_inner_slice, dim3(g, g, size.z-g  ));
        req[3].map.set(xy_inner_slice, dim3(g, g,        g-1));
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], 1, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
        CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[2], 1, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
        CPM::cpm_irecv_colored(           domain, req[3], 1, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
    }
    if (mpi.rank == 0) {
        printf("communicating color 1...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else if (mpi.rank == mpi.size - 1) {
        CPM::cpm_waitall(req, 2, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], 1, LOC::DEVICE);
    } else {
        CPM::cpm_waitall(req, 4, MPI_STATUSES_IGNORE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], 1, LOC::DEVICE);
        CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], 1, LOC::DEVICE);
    }
    for (int i = 0; i < 4; i ++) {
        req[i].release();
    }
    x.sync(SYNC::D2H);
    for (int rank = 0; rank < mpi.size; rank ++) {
        if (rank == mpi.rank) {
            printf("%d printing...\n", rank);
            print_slice(x, domain, XZ_PLAIN, slice_at);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}