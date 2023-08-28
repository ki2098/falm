#ifndef _POI_H_
#define _POI_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include "StructuredField.cuh"
#include "StructuredMesh.cuh"
#include "Util.cuh"

namespace FALMOld {

struct LS_State {
    double re;
    int    it;
};

__global__ static void poisson_sor_kernel(FieldCp<double> &a, FieldCp<double> &x, FieldCp<double> &b, double omega, int color, Dom dom, Dom mapper) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::d321(i  , j  , k  , dom._size);
        o1 = FALMUtil::d321(i+1, j  , k  , dom._size);
        o2 = FALMUtil::d321(i-1, j  , k  , dom._size);
        o3 = FALMUtil::d321(i  , j+1, k  , dom._size);
        o4 = FALMUtil::d321(i  , j-1, k  , dom._size);
        o5 = FALMUtil::d321(i  , j  , k+1, dom._size);
        o6 = FALMUtil::d321(i  , j  , k-1, dom._size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(o0);
        x1 = x(o1);
        x2 = x(o2);
        x3 = x(o3);
        x4 = x(o4);
        x5 = x(o5);
        x6 = x(o6);
        double c0 = (b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        if ((i + j + k + dom._offset.x + dom._offset.y + dom._offset.z) % 2 != color) {
            c0 = 0;
        }
        x(o0) = x0 + omega * c0;
    }
}

__global__ static void poisson_jacobi_kernel(FieldCp<double> &a, FieldCp<double> &xn, FieldCp<double> &xp, FieldCp<double> &b, Dom dom, Dom mapper) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::d321(i  , j  , k  , dom._size);
        o1 = FALMUtil::d321(i+1, j  , k  , dom._size);
        o2 = FALMUtil::d321(i-1, j  , k  , dom._size);
        o3 = FALMUtil::d321(i  , j+1, k  , dom._size);
        o4 = FALMUtil::d321(i  , j-1, k  , dom._size);
        o5 = FALMUtil::d321(i  , j  , k+1, dom._size);
        o6 = FALMUtil::d321(i  , j  , k-1, dom._size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = xp(o0);
        x1 = xp(o1);
        x2 = xp(o2);
        x3 = xp(o3);
        x4 = xp(o4);
        x5 = xp(o5);
        x6 = xp(o6);
        double c0 = (b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6)) / a0;
        xn(o0) = x0 + c0;
    }
}

__global__ static void res_kernel(FieldCp<double> &a, FieldCp<double> &x, FieldCp<double> &b, FieldCp<double> &r, Dom dom, Dom mapper) {
    unsigned int stride = FALMUtil::get_global_size();
    for (unsigned int idx = FALMUtil::get_global_idx(); idx < mapper._num; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, mapper._size);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int o0, o1, o2, o3, o4, o5, o6;
        o0 = FALMUtil::d321(i  , j  , k  , dom._size);
        o1 = FALMUtil::d321(i+1, j  , k  , dom._size);
        o2 = FALMUtil::d321(i-1, j  , k  , dom._size);
        o3 = FALMUtil::d321(i  , j+1, k  , dom._size);
        o4 = FALMUtil::d321(i  , j-1, k  , dom._size);
        o5 = FALMUtil::d321(i  , j  , k+1, dom._size);
        o6 = FALMUtil::d321(i  , j  , k-1, dom._size);
        double a0, a1, a2, a3, a4, a5, a6;
        a0 = a(o0, 0);
        a1 = a(o0, 1);
        a2 = a(o0, 2);
        a3 = a(o0, 3);
        a4 = a(o0, 4);
        a5 = a(o0, 5);
        a6 = a(o0, 6);
        double x0, x1, x2, x3, x4, x5, x6;
        x0 = x(o0);
        x1 = x(o1);
        x2 = x(o2);
        x3 = x(o3);
        x4 = x(o4);
        x5 = x(o5);
        x6 = x(o6);
        r(o0) = b(o0) - (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 + a5 * x5 + a6 * x6);
    }
}

static void poisson_sor(Field<double> &a, Field<double> &x, Field<double> &b, Field<double> &r, double omega, int maxit, Dom &dom, Dom &global, int mpi_size, int mpi_rank, MPI_Request *req, LS_State &state, double &kernel_time, double &comm_time) {
    dim3 &osz = dom._size;
    const unsigned int g  = guide;
    const unsigned int g2 = 2 * guide;
    int color;
    double err;
    double t0, t1, t2;
    Dom map;
    for (state.it = 0; state.it < maxit;) {
        color = 0;
        if (mpi_size > 1) {
            int buflen = osz.y * osz.z;
            if (mpi_rank == 0) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[1]);
            } else {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[3]);
            }
        }
        // map needed
        t0 = MPI_Wtime();
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
        t1 = MPI_Wtime();
        if (mpi_size > 1) {
            if (mpi_rank == 0) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            } else {
                MPI_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            }
        }
        kernel_time += (t1 - t0);
        comm_time   += (t2 - t0);

        color = 1;
        if (mpi_size > 1) {
            int buflen = osz.y * osz.z;
            if (mpi_rank == 0) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[1]);
            } else {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[3]);
            }
        }
        // map needed
        t0 = MPI_Wtime();
        poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
        t1 = MPI_Wtime();
        if (mpi_size > 1) {
            if (mpi_rank == 0) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            } else {
                MPI_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
                // map needed
                poisson_sor_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), sor_omega, color, dom, map);
            }
        }
        kernel_time += (t1 - t0);
        comm_time   += (t2 - t0);

        // map needed
        res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), dom, map);
        err = FALMUtil::fscala_norm2(r, dom);
        if (mpi_size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        dim3 &gsz = global._size;
        unsigned int gnum = (gsz.x - 2 * g) * (gsz.y - 2 * g) * (gsz.z - 2 * g);
        state.re = sqrt(err) / gnum;
        state.it ++;
    } while (state.it < maxit && state.re > ls_epsilon);
}

static void poisson_jacobi(Field<double> &a, Field<double> &x, Field<double> &xp, Field<double> &b, Field<double> &r, double omega, int maxit, Dom &dom, Dom& global, int mpi_size, int mpi_rank, MPI_Request *req, LS_State &state, double &kernel_time, double &comm_time) {
    dim3 &osz = dom._size;
    const unsigned int g  = guide;
    const unsigned int g2 = 2 * guide;
    double err;
    double t0, t1, t2;
    unsigned int i_head, i_tail;
    Dom map;
    for (state.it = 0; state.it < maxit;) {
        FALMUtil::field_cpy(xp, x, FALMLoc::HOST);
        if (mpi_size > 1) {
            int buflen = osz.y * osz.z;
            if (mpi_rank == 0) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[1]);
            } else {
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(osz.x-g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[0]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(osz.x-g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[1]);
                MPI_Isend(&(x._hd._arr[FALMUtil::d321(      g  ,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[2]);
                MPI_Irecv(&(x._hd._arr[FALMUtil::d321(      g-1,0,0,osz)]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[3]);
            }
        }
        // map needed
        t0 = MPI_Wtime();
        poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), dom, map);
        t1 = MPI_Wtime();
        if (mpi_size > 1) {
            if (mpi_rank == 0) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), dom, map);
            } else if (mpi_rank == mpi_size - 1) {
                MPI_Waitall(2, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), dom, map);
            } else {
                MPI_Waitall(4, &req[0], MPI_STATUSES_IGNORE);
                t2 = MPI_Wtime();
                // map needed
                poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), dom, map);
                // map needed
                poisson_jacobi_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(xp._dd), *(b._dd), dom, map);
            }
        }
        kernel_time += (t1 - t0);
        comm_time   += (t2 - t0);

        // map needed
        res_kernel<<<n_blocks, n_threads>>>(*(a._dd), *(x._dd), *(b._dd), *(r._dd), dom, map);
        err = FALMUtil::fscala_norm2(r, dom);
        if (mpi_size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        dim3 &gsz = global._size;
        unsigned int gnum = (gsz.x - 2 * g) * (gsz.y - 2 * g) * (gsz.z - 2 * g);
        state.re = sqrt(err) / gnum;
        state.it ++;
    } while (state.it < maxit && state.re > ls_epsilon);
}



}

#endif