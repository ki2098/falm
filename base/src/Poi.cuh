#ifndef FALM_POI_CUH
#define FALM_POI_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include "param.h"
#include "Util.cuh"
#include "Field.cuh"
#include "Mapper.cuh"
#include "CPM.cuh"

namespace FALM {

const unsigned int BLACK = 0;
const unsigned int RED   = 1;

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(FieldFrame<double> &a, FieldFrame<double> &x, FieldFrame<double> &b, unsigned int color, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int id0, id1, id2, id3, id4, id5, id6;
        id0 = UTIL::IDX(i  , j  , k  , size);
        id1 = UTIL::IDX(i+1, j  , k  , size);
        id2 = UTIL::IDX(i-1, j  , k  , size);
        id3 = UTIL::IDX(i  , j+1, k  , size);
        id4 = UTIL::IDX(i  , j-1, k  , size);
        id5 = UTIL::IDX(i  , j  , k+1, size);
        id6 = UTIL::IDX(i  , j  , k-1, size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(id0, 0);
        a1 = a(id0, 1);
        a2 = a(id0, 2);
        a3 = a(id0, 3);
        a4 = a(id0, 4);
        a5 = a(id0, 5);
        a6 = a(id0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(id0);
        x1 = x(id1);
        x2 = x(id2);
        x3 = x(id3);
        x4 = x(id4);
        x5 = x(id5);
        x6 = x(id6);
        double c0 = (b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        if ((i + j + k + origin.x + origin.y + origin.z) % 2 != color) {
            c0 = 0;
        }
        x(id0) = x0 + sor_omega * c0;
    }
}

__global__ void poisson_jacobi_kernel(FieldFrame<double> &a, FieldFrame<double> &xn, FieldFrame<double> &xp, FieldFrame<double> &b, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int id0, id1, id2, id3, id4, id5, id6;
        id0 = UTIL::IDX(i  , j  , k  , size);
        id1 = UTIL::IDX(i+1, j  , k  , size);
        id2 = UTIL::IDX(i-1, j  , k  , size);
        id3 = UTIL::IDX(i  , j+1, k  , size);
        id4 = UTIL::IDX(i  , j-1, k  , size);
        id5 = UTIL::IDX(i  , j  , k+1, size);
        id6 = UTIL::IDX(i  , j  , k-1, size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(id0, 0);
        a1 = a(id0, 1);
        a2 = a(id0, 2);
        a3 = a(id0, 3);
        a4 = a(id0, 4);
        a5 = a(id0, 5);
        a6 = a(id0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = xp(id0);
        x1 = xp(id1);
        x2 = xp(id2);
        x3 = xp(id3);
        x4 = xp(id4);
        x5 = xp(id5);
        x6 = xp(id6);
        double c0 = (b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        xn(id0) = x0 + c0;
    }
}

__global__ static void res_kernel(FieldFrame<double> &a, FieldFrame<double> &x, FieldFrame<double> &b, FieldFrame<double> &r, Mapper domain, Mapper range) {
    unsigned int i, j, k;
    UTIL::THREAD2IJK(i, j, k);
    dim3 &size = domain.size;
    dim3 &origin = domain.offset;
    if (i < range.size.x && j < range.size.y && k < range.size.z) {
        i += range.offset.x;
        j += range.offset.y;
        k += range.offset.z;
        unsigned int id0, id1, id2, id3, id4, id5, id6;
        id0 = UTIL::IDX(i  , j  , k  , size);
        id1 = UTIL::IDX(i+1, j  , k  , size);
        id2 = UTIL::IDX(i-1, j  , k  , size);
        id3 = UTIL::IDX(i  , j+1, k  , size);
        id4 = UTIL::IDX(i  , j-1, k  , size);
        id5 = UTIL::IDX(i  , j  , k+1, size);
        id6 = UTIL::IDX(i  , j  , k-1, size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(id0, 0);
        a1 = a(id0, 1);
        a2 = a(id0, 2);
        a3 = a(id0, 3);
        a4 = a(id0, 4);
        a5 = a(id0, 5);
        a6 = a(id0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(id0);
        x1 = x(id1);
        x2 = x(id2);
        x3 = x(id3);
        x4 = x(id4);
        x5 = x(id5);
        x6 = x(id6);
        r(id0) = b(id0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6);
    }
}

static void poisson_sor_color_phase(Field<double> &a, Field<double> &x, Field<double> &b, Field<double> &r, Mapper &domain, Mapper &global, MPI_State &mpi, CPM::Request *req, LS_State &state, double &kernel_time, double &comm_time, unsigned int color) {
    dim3 &size = domain.size;
    const unsigned int g = guide;
    double t0, t1, t2;
    Mapper map;
    unsigned int __color = UTIL::flip_color(color);

    if (mpi.size > 1) {
        dim3 yz_inner_slice(1, size.y - 2 * g, size.z - 2 * g);
        if (mpi.rank == 0) {
            req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
            req[1].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
            CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], __color, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
            CPM::cpm_irecv_colored(           domain, req[1], __color, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
            map.set(
                dim3(size.x - 2 * g - 1, size.y - 2 * g, size.z - 2 * g),
                dim3(g, g, g)
            );
        } else if (mpi.rank == mpi.size - 1) {
            req[0].map.set(yz_inner_slice, dim3(       g  , g, g));
            req[1].map.set(yz_inner_slice, dim3(       g-1, g, g));
            CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], __color, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
            CPM::cpm_irecv_colored(           domain, req[1], __color, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
            map.set(
                dim3(size.x - 2 * g - 1, size.y - 2 * g, size.z - 2 * g),
                dim3(g + 1, g, g)
            );
        } else {
            req[0].map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
            req[1].map.set(yz_inner_slice, dim3(       g  , g, g));
            req[2].map.set(yz_inner_slice, dim3(size.x-g  , g, g));
            req[3].map.set(yz_inner_slice, dim3(       g-1, g, g));
            CPM::cpm_isend_colored(x.dev.ptr, domain, req[0], __color, LOC::DEVICE, mpi.rank+1, 0, MPI_COMM_WORLD);
            CPM::cpm_isend_colored(x.dev.ptr, domain, req[1], __color, LOC::DEVICE, mpi.rank-1, 1, MPI_COMM_WORLD);
            CPM::cpm_irecv_colored(           domain, req[2], __color, LOC::DEVICE, mpi.rank+1, 1, MPI_COMM_WORLD);
            CPM::cpm_irecv_colored(           domain, req[3], __color, LOC::DEVICE, mpi.rank-1, 0, MPI_COMM_WORLD);
            map.set(
                dim3(size.x - 2 * g - 2, size.y - 2 * g, size.z - 2 * g),
                dim3(g + 1, g, g)
            );
        }
    } else {
        map.set(
            dim3(size.x - 2 * g, size.y - 2 * g, size.z - 2 * g),
            dim3(g, g, g)
        );
    }

    dim3 grid_size(
        (map.size.x + block_size.x - 1) / block_size.x,
        (map.size.y + block_size.y - 1) / block_size.y,
        (map.size.z + block_size.z - 1) / block_size.z
    );
    t0 = MPI_Wtime();
    poisson_sor_kernel<<<grid_size, block_size>>>(*(a.devptr), *(x.devptr), *(b.devptr), color, domain, map);
    t1 = MPI_Wtime();

    if (mpi.size > 1) {
        dim3 yz_inner_slice(1, size.y - 2 * g, size.z - 2 * g);
        if (mpi.rank == 0) {
            CPM::cpm_waitall(req, 2);
            CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], __color, LOC::DEVICE);
            t2 = MPI_Wtime();
            map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
            dim3 block(
                min(yz_inner_slice.x, block_size.x), 
                min(yz_inner_slice.y, block_size.y), 
                min(yz_inner_slice.z, block_size.z)
            ); 
            dim3 grid(
                (yz_inner_slice.x + block.x - 1) / block.x,
                (yz_inner_slice.y + block.y - 1) / block.y,
                (yz_inner_slice.z + block.z - 1) / block.z
            );
            poisson_sor_kernel<<<grid, block>>>(*(a.devptr), *(x.devptr), *(b.devptr), color, domain, map);
        } else if (mpi.rank == mpi.size - 1) {
            CPM::cpm_waitall(req, 2);
            CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[1], __color, LOC::DEVICE);
            t2 = MPI_Wtime();
            map.set(yz_inner_slice, dim3(g, g, g));
            dim3 block(
                min(yz_inner_slice.x, block_size.x), 
                min(yz_inner_slice.y, block_size.y), 
                min(yz_inner_slice.z, block_size.z)
            ); 
            dim3 grid(
                (yz_inner_slice.x + block.x - 1) / block.x,
                (yz_inner_slice.y + block.y - 1) / block.y,
                (yz_inner_slice.z + block.z - 1) / block.z
            );
            poisson_sor_kernel<<<grid, block>>>(*(a.devptr), *(x.devptr), *(b.devptr), color, domain, map);
        } else {
            CPM::cpm_waitall(req, 4);
            CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[2], __color, LOC::DEVICE);
            CPM::cpm_unpack_buffer_colored(x.dev.ptr, domain, req[3], __color, LOC::DEVICE);
            dim3 block(
                min(yz_inner_slice.x, block_size.x), 
                min(yz_inner_slice.y, block_size.y), 
                min(yz_inner_slice.z, block_size.z)
            ); 
            dim3 grid(
                (yz_inner_slice.x + block.x - 1) / block.x,
                (yz_inner_slice.y + block.y - 1) / block.y,
                (yz_inner_slice.z + block.z - 1) / block.z
            );
            map.set(yz_inner_slice, dim3(size.x-g-1, g, g));
            poisson_sor_kernel<<<grid, block>>>(*(a.devptr), *(x.devptr), *(b.devptr), color, domain, map);
            map.set(yz_inner_slice, dim3(g, g, g));
            poisson_sor_kernel<<<grid, block>>>(*(a.devptr), *(x.devptr), *(b.devptr), color, domain, map);
        }
    }

    kernel_time += (t1 - t0);
    comm_time   += (t2 - t0);
}

static void poisson_sor(Field<double> &a, Field<double> &x, Field<double> &b, Field<double> &r, Mapper &domain, Mapper &global, MPI_State &mpi, CPM::Request *req, LS_State &state, double &kernel_time, double &comm_time) {
    dim3 &size = domain.size;
    unsigned int g = guide;
    unsigned int color;
    double err;
    double t0, t1, t2;
    state.it = 0;
    Mapper map;
    int mpi_size = mpi.size;
    int mpi_rank = mpi.rank;
    do {
        poisson_sor_color_phase(a, x, b, r, domain, global, mpi, req, state, kernel_time, comm_time, BLACK);
        poisson_sor_color_phase(a, x, b, r, domain, global, mpi, req, state, kernel_time, comm_time, RED  );

    } while (state.it < ls_maxit && state.re > ls_epsilon);
}

}

#endif