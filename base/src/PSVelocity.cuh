#ifndef _PSVELOCITY_H_
#define _PSVELOCITY_H_ 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <mpi.h>
#include "StructuredField.cuh"
#include "Util.cuh"

namespace FALM {

__device__ static double upwind3(double u00, double u11, double u12, double u13, double u14, double u21, double u22, double u23, double u24, double u31, double u32, double u33, double u34, double abs1, double abs2, double abs3, double uu11, double uu12, double uu21, double uu22, double uu31, double uu32, double j00) {
    double adv = 0;
    double j2  = 2 * j00;
    adv += uu12 * (- u14 + 27 * u13 - 27 * u00 + u12) / j2;
    adv += uu11 * (- u13 + 27 * u00 - 27 * u12 + u11) / j2;
    adv += abs1 * (u14 - 4 * u13 + 6 * u00 - 4 * u12 + u11);
    adv += uu22 * (- u24 + 27 * u23 - 27 * u00 + u22) / j2;
    adv += uu21 * (- u23 + 27 * u00 - 27 * u22 + u21) / j2;
    adv += abs2 * (u24 - 4 * u23 + 6 * u00 - 4 * u22 + u21);
    adv += uu32 * (- u34 + 27 * u33 - 27 * u00 + u32) / j2;
    adv += uu31 * (- u33 + 27 * u00 - 27 * u32 + u31) / j2;
    adv += abs3 * (u34 - 4 * u33 + 6 * u00 - 4 * u32 + u31);
    adv /= 24.0;
    return adv;
}

__device__ static double diffusion(double u00, double u11, double u12, double u21, double u22, double u31, double u32, double n00, double n11, double n12, double n21, double n22, double n31, double n32, double g10, double g11, double g12,double g20, double g21, double g22, double g30, double g31, double g32, double j00) {
    double vis = 0;
    vis += (g12 + g10) * (ri + 0.5 * (n00 + n12)) * (u12 - u00);
    vis -= (g11 + g10) * (ri + 0.5 * (n00 + n11)) * (u00 - u11);
    vis += (g22 + g20) * (ri + 0.5 * (n00 + n22)) * (u22 - u00);
    vis -= (g21 + g20) * (ri + 0.5 * (n00 + n21)) * (u00 - u21);
    vis += (g32 + g30) * (ri + 0.5 * (n00 + n32)) * (u32 - u00);
    vis -= (g31 + g30) * (ri + 0.5 * (n00 + n31)) * (u00 - u31);
    vis /= (2 * j00);
    return vis;
}

__global__ static void psvelocity_kernel(FieldCp<double> &u, FieldCp<double> &uu, FieldCp<double> &ua, FieldCp<double> &nut, FieldCp<double> &kx, FieldCp<double> &g, FieldCp<double> &ja, FieldCp<double> &ff, DomCp &dom, DomCp &mapper, unsigned int idx_start, unsigned int idx_end) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz =    dom._size;
    dim3 &osz = mapper._size;
    for (unsigned int idx = FALMUtil::get_global_idx() + idx_start; idx < idx_end; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, isz);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int o00;
        unsigned int o11, o12, o13, o14;
        unsigned int o21, o22, o23, o24;
        unsigned int o31, o32, o33, o34;
        o00 = FALMUtil::d321(i  ,j  ,k  ,osz);
        o11 = FALMUtil::d321(i-2,j  ,k  ,osz);
        o12 = FALMUtil::d321(i-1,j  ,k  ,osz);
        o13 = FALMUtil::d321(i+1,j  ,k  ,osz);
        o14 = FALMUtil::d321(i+2,j  ,k  ,osz);
        o21 = FALMUtil::d321(i  ,j-2,k  ,osz);
        o22 = FALMUtil::d321(i  ,j-1,k  ,osz);
        o23 = FALMUtil::d321(i  ,j+1,k  ,osz);
        o24 = FALMUtil::d321(i  ,j+2,k  ,osz);
        o31 = FALMUtil::d321(i  ,j  ,k-2,osz);
        o32 = FALMUtil::d321(i  ,j  ,k-1,osz);
        o33 = FALMUtil::d321(i  ,j  ,k+1,osz);
        o34 = FALMUtil::d321(i  ,j  ,k+2,osz);
        double abs1 = fabs(u(o00,0) * kx(o00,0));
        double abs2 = fabs(u(o00,1) * kx(o00,1));
        double abs3 = fabs(u(o00,2) * kx(o00,2));
        double uu11 =  uu(o12,0);
        double uu12 =  uu(o00,0);
        double uu21 =  uu(o22,1);
        double uu22 =  uu(o00,1);
        double uu31 =  uu(o32,2);
        double uu32 =  uu(o00,2);
        double  n00 = nut(o00);
        double  n11 = nut(o12);
        double  n12 = nut(o13);
        double  n21 = nut(o22);
        double  n22 = nut(o23);
        double  n31 = nut(o32);
        double  n32 = nut(o33);
        double  g10 =   g(o00,0);
        double  g11 =   g(o12,0);
        double  g12 =   g(o13,0);
        double  g20 =   g(o00,1);
        double  g21 =   g(o22,1);
        double  g22 =   g(o23,1);
        double  g30 =   g(o00,2);
        double  g31 =   g(o32,2);
        double  g32 =   g(o33,2);
        double  j00 =  ja(o00);
        unsigned int m;
        double u00;
        double u11, u12, u13, u14;
        double u21, u22, u23, u24;
        double u31, u32, u33, u34;
        double adv, vis;
        
        m = 0;
        u00 = u(o00,m);
        u11 = u(o11,m);
        u12 = u(o12,m);
        u13 = u(o13,m);
        u14 = u(o14,m);
        u21 = u(o21,m);
        u22 = u(o22,m);
        u23 = u(o23,m);
        u24 = u(o24,m);
        u31 = u(o31,m);
        u32 = u(o32,m);
        u33 = u(o33,m);
        u34 = u(o34,m);
        adv = upwind3(u00, u11, u12, u13, u14, u21, u22, u23, u24, u31, u32, u33, u34, abs1, abs2, abs3, uu11, uu12, uu21, uu22, uu31, uu32, j00);
        vis = diffusion(u00, u12, u13, u22, u23, u32, u33, n00, n11, n12, n21, n22, n31, n32, g10, g11, g12, g20, g21, g22, g30, g31, g32, j00);
        ua(o00,m) = u00 + dt * (- adv + vis + ff(o00,m));

        m = 1;
        u00 = u(o00,m);
        u11 = u(o11,m);
        u12 = u(o12,m);
        u13 = u(o13,m);
        u14 = u(o14,m);
        u21 = u(o21,m);
        u22 = u(o22,m);
        u23 = u(o23,m);
        u24 = u(o24,m);
        u31 = u(o31,m);
        u32 = u(o32,m);
        u33 = u(o33,m);
        u34 = u(o34,m);
        adv = upwind3(u00, u11, u12, u13, u14, u21, u22, u23, u24, u31, u32, u33, u34, abs1, abs2, abs3, uu11, uu12, uu21, uu22, uu31, uu32, j00);
        vis = diffusion(u00, u12, u13, u22, u23, u32, u33, n00, n11, n12, n21, n22, n31, n32, g10, g11, g12, g20, g21, g22, g30, g31, g32, j00);
        ua(o00,m) = u00 + dt * (- adv + vis + ff(o00,m));

        m = 2;
        u00 = u(o00,m);
        u11 = u(o11,m);
        u12 = u(o12,m);
        u13 = u(o13,m);
        u14 = u(o14,m);
        u21 = u(o21,m);
        u22 = u(o22,m);
        u23 = u(o23,m);
        u24 = u(o24,m);
        u31 = u(o31,m);
        u32 = u(o32,m);
        u33 = u(o33,m);
        u34 = u(o34,m);
        adv = upwind3(u00, u11, u12, u13, u14, u21, u22, u23, u24, u31, u32, u33, u34, abs1, abs2, abs3, uu11, uu12, uu21, uu22, uu31, uu32, j00);
        vis = diffusion(u00, u12, u13, u22, u23, u32, u33, n00, n11, n12, n21, n22, n31, n32, g10, g11, g12, g20, g21, g22, g30, g31, g32, j00);
        ua(o00,m) = u00 + dt * (- adv + vis + ff(o00,m));
    }
}

static void psvelocity(Field<double> &u, Field<double> &uu, Field<double> &ua, Field<double> &nut, Field<double> &kx, Field<double> &gm, Field<double> &ja, Field<double> &ff, Dom &dom, Dom &global, Dom &inner, int mpi_size, int mpi_rank, MPI_Request *req, double &kernel_time, double &comm_time) {
    dim3 &osz =   dom._h._size;
    dim3 &isz = inner._h._size;
    dim3 &ift = inner._h._offset;
    unsigned int idx_start, idx_end;
    unsigned int i_start, i_end;
    double t0, t1, t2;
    if (mpi_size > 1) {
        unsigned int buflen = osz.y * osz.z;
        unsigned int  dlen1 = dom._h._num;
        unsigned int  dlen2 = 2 * dlen1;
        unsigned int  send0 = FALMUtil::d321(      ift.x  ,0,0,osz);
        unsigned int  send1 = FALMUtil::d321(      ift.x+1,0,0,osz);
        unsigned int  send2 = FALMUtil::d321(isz.x+ift.x-2,0,0,osz);
        unsigned int  send3 = FALMUtil::d321(isz.x+ift.x-1,0,0,osz);
        unsigned int  recv0 = FALMUtil::d321(      ift.x-2,0,0,osz);
        unsigned int  recv1 = FALMUtil::d321(      ift.x-1,0,0,osz);
        unsigned int  recv2 = FALMUtil::d321(isz.x+ift.x  ,0,0,osz);
        unsigned int  recv3 = FALMUtil::d321(isz.x+ift.x+1,0,0,osz);
        if (mpi_rank == 0) {
            MPI_Isend(&(  u._hd._arr[send2      ]), buflen, MPI_DOUBLE, mpi_rank+1, 4, MPI_COMM_WORLD, &req[ 0]);
            MPI_Isend(&(  u._hd._arr[send2+dlen1]), buflen, MPI_DOUBLE, mpi_rank+1, 5, MPI_COMM_WORLD, &req[ 1]);
            MPI_Isend(&(  u._hd._arr[send2+dlen2]), buflen, MPI_DOUBLE, mpi_rank+1, 6, MPI_COMM_WORLD, &req[ 2]);
            MPI_Isend(&(nut._hd._arr[send3      ]), buflen, MPI_DOUBLE, mpi_rank+1, 7, MPI_COMM_WORLD, &req[ 3]);
            MPI_Irecv(&(  u._hd._arr[recv3      ]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[ 4]);
            MPI_Irecv(&(  u._hd._arr[recv3+dlen1]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[ 5]);
            MPI_Irecv(&(  u._hd._arr[recv3+dlen2]), buflen, MPI_DOUBLE, mpi_rank+1, 2, MPI_COMM_WORLD, &req[ 6]);
            MPI_Irecv(&(nut._hd._arr[recv2      ]), buflen, MPI_DOUBLE, mpi_rank+1, 3, MPI_COMM_WORLD, &req[ 7]);
        } else if (mpi_rank == mpi_size - 1) {
            MPI_Isend(&(  u._hd._arr[send1      ]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[ 0]);
            MPI_Isend(&(  u._hd._arr[send1+dlen1]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[ 1]);
            MPI_Isend(&(  u._hd._arr[send1+dlen2]), buflen, MPI_DOUBLE, mpi_rank-1, 2, MPI_COMM_WORLD, &req[ 2]);
            MPI_Isend(&(nut._hd._arr[send0      ]), buflen, MPI_DOUBLE, mpi_rank-1, 3, MPI_COMM_WORLD, &req[ 3]);
            MPI_Irecv(&(  u._hd._arr[recv0      ]), buflen, MPI_DOUBLE, mpi_rank-1, 4, MPI_COMM_WORLD, &req[ 4]);
            MPI_Irecv(&(  u._hd._arr[recv0+dlen1]), buflen, MPI_DOUBLE, mpi_rank-1, 5, MPI_COMM_WORLD, &req[ 5]);
            MPI_Irecv(&(  u._hd._arr[recv0+dlen2]), buflen, MPI_DOUBLE, mpi_rank-1, 6, MPI_COMM_WORLD, &req[ 6]);
            MPI_Irecv(&(nut._hd._arr[recv1      ]), buflen, MPI_DOUBLE, mpi_rank-1, 7, MPI_COMM_WORLD, &req[ 7]);
        } else {
            MPI_Isend(&(  u._hd._arr[send2      ]), buflen, MPI_DOUBLE, mpi_rank+1, 4, MPI_COMM_WORLD, &req[ 0]);
            MPI_Isend(&(  u._hd._arr[send2+dlen1]), buflen, MPI_DOUBLE, mpi_rank+1, 5, MPI_COMM_WORLD, &req[ 1]);
            MPI_Isend(&(  u._hd._arr[send2+dlen2]), buflen, MPI_DOUBLE, mpi_rank+1, 6, MPI_COMM_WORLD, &req[ 2]);
            MPI_Isend(&(nut._hd._arr[send3      ]), buflen, MPI_DOUBLE, mpi_rank+1, 7, MPI_COMM_WORLD, &req[ 3]);
            MPI_Isend(&(  u._hd._arr[send1      ]), buflen, MPI_DOUBLE, mpi_rank-1, 0, MPI_COMM_WORLD, &req[ 4]);
            MPI_Isend(&(  u._hd._arr[send1+dlen1]), buflen, MPI_DOUBLE, mpi_rank-1, 1, MPI_COMM_WORLD, &req[ 5]);
            MPI_Isend(&(  u._hd._arr[send1+dlen2]), buflen, MPI_DOUBLE, mpi_rank-1, 2, MPI_COMM_WORLD, &req[ 6]);
            MPI_Isend(&(nut._hd._arr[send0      ]), buflen, MPI_DOUBLE, mpi_rank-1, 3, MPI_COMM_WORLD, &req[ 7]);

            MPI_Irecv(&(  u._hd._arr[recv3      ]), buflen, MPI_DOUBLE, mpi_rank+1, 0, MPI_COMM_WORLD, &req[ 8]);
            MPI_Irecv(&(  u._hd._arr[recv3+dlen1]), buflen, MPI_DOUBLE, mpi_rank+1, 1, MPI_COMM_WORLD, &req[ 9]);
            MPI_Irecv(&(  u._hd._arr[recv3+dlen2]), buflen, MPI_DOUBLE, mpi_rank+1, 2, MPI_COMM_WORLD, &req[10]);
            MPI_Irecv(&(nut._hd._arr[recv2      ]), buflen, MPI_DOUBLE, mpi_rank+1, 3, MPI_COMM_WORLD, &req[11]);
            MPI_Irecv(&(  u._hd._arr[recv0      ]), buflen, MPI_DOUBLE, mpi_rank-1, 4, MPI_COMM_WORLD, &req[12]);
            MPI_Irecv(&(  u._hd._arr[recv0+dlen1]), buflen, MPI_DOUBLE, mpi_rank-1, 5, MPI_COMM_WORLD, &req[13]);
            MPI_Irecv(&(  u._hd._arr[recv0+dlen2]), buflen, MPI_DOUBLE, mpi_rank-1, 6, MPI_COMM_WORLD, &req[14]);
            MPI_Irecv(&(nut._hd._arr[recv1      ]), buflen, MPI_DOUBLE, mpi_rank-1, 7, MPI_COMM_WORLD, &req[15]);
        }
    }

    i_start   = (mpi_rank > 0)? 2 : 0;
    i_end     = (mpi_rank == mpi_size - 1)? isz.x : isz.x-2;
    idx_start = FALMUtil::d321(i_start,0,0,isz);
    idx_end   = FALMUtil::d321(i_end  ,0,0,isz);
    t0 = MPI_Wtime();
    psvelocity_kernel<<<n_blocks, n_threads>>>(*(u._dd), *(uu._dd), *(ua._dd), *(nut._dd), *(kx._dd), *(gm._dd), *(ja._dd), *(ff._dd), *(dom._d), *(inner._d), idx_start, idx_end);
    t1 = MPI_Wtime();

    if (mpi_size > 1) {
        if (mpi_rank == 0) {
            MPI_Waitall(4, &req[4], MPI_STATUSES_IGNORE);
            t2 = MPI_Wtime();
            idx_start = FALMUtil::d321(isz.x-2,0,0,isz);
            idx_end   = FALMUtil::d321(isz.x  ,0,0,isz);
            psvelocity_kernel<<<n_blocks, n_threads>>>(*(u._dd), *(uu._dd), *(ua._dd), *(nut._dd), *(kx._dd), *(gm._dd), *(ja._dd), *(ff._dd), *(dom._d), *(inner._d), idx_start, idx_end);
        } else if (mpi_rank == mpi_size - 1) {
            MPI_Waitall(4, &req[4], MPI_STATUSES_IGNORE);
            t2 = MPI_Wtime();
            idx_start = FALMUtil::d321(      0,0,0,isz);
            idx_end   = FALMUtil::d321(      2,0,0,isz);
            psvelocity_kernel<<<n_blocks, n_threads>>>(*(u._dd), *(uu._dd), *(ua._dd), *(nut._dd), *(kx._dd), *(gm._dd), *(ja._dd), *(ff._dd), *(dom._d), *(inner._d), idx_start, idx_end);
        } else {
            MPI_Waitall(8, &req[8], MPI_STATUSES_IGNORE);
            t2 = MPI_Wtime();
            idx_start = FALMUtil::d321(isz.x-2,0,0,isz);
            idx_end   = FALMUtil::d321(isz.x  ,0,0,isz);
            psvelocity_kernel<<<n_blocks, n_threads>>>(*(u._dd), *(uu._dd), *(ua._dd), *(nut._dd), *(kx._dd), *(gm._dd), *(ja._dd), *(ff._dd), *(dom._d), *(inner._d), idx_start, idx_end);
            idx_start = FALMUtil::d321(      0,0,0,isz);
            idx_end   = FALMUtil::d321(      2,0,0,isz);
            psvelocity_kernel<<<n_blocks, n_threads>>>(*(u._dd), *(uu._dd), *(ua._dd), *(nut._dd), *(kx._dd), *(gm._dd), *(ja._dd), *(ff._dd), *(dom._d), *(inner._d), idx_start, idx_end);
        }
    }
    kernel_time += (t1 - t0);
    comm_time   += (t2 - t0);
}

__global__ static void interp_psvelocity_kernel_1(FieldCp<double> &u, FieldCp<double> &uc, FieldCp<double> &kx, FieldCp<double> &ja, DomCp &dom, DomCp &mapper, unsigned int idx_start, unsigned int idx_end) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = mapper._size;
    dim3 &osz =    dom._size;
    for (unsigned int idx = FALMUtil::get_global_idx() + idx_start; idx < idx_end; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, isz);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int odx = FALMUtil::d321(i, j, k, osz);
        double jo = ja(odx);
        uc(odx,0) = jo * kx(odx,0) * u(odx,0);
        uc(odx,1) = jo * kx(odx,1) * u(odx,1);
        uc(odx,2) = jo * kx(odx,2) * u(odx,2);
    }
}

__global__ static void interp_psvelocity_kernel_2(FieldCp<double> &uc, FieldCp<double> &uu, DomCp &dom, DomCp &mapper, unsigned int idx_start, unsigned int idx_end) {
    unsigned int stride = FALMUtil::get_global_size();
    dim3 &isz = mapper._size;
    dim3 &osz =    dom._size;
    for (unsigned int idx = FALMUtil::get_global_idx() + idx_start; idx < idx_end; idx += stride) {
        unsigned int ii, ij, ik;
        FALMUtil::d123(idx, ii, ij, ik, isz);
        unsigned int i, j, k;
        i = ii + mapper._offset.x;
        j = ij + mapper._offset.y;
        k = ik + mapper._offset.z;
        unsigned int odx = FALMUtil::d321(i, j, k, osz);
        uu(odx,0) = 0.5 * (uc(odx,0) + uc(FALMUtil::d321(i+1,j,k,osz),0));
        uu(odx,1) = 0.5 * (uc(odx,1) + uc(FALMUtil::d321(i,j+1,k,osz),1));
        uu(odx,2) = 0.5 * (uc(odx,2) + uc(FALMUtil::d321(i,j,k+1,osz),2));
    }
}

static void interp_psvelocity(Field<double> &u, Field<double> &uc, Field<double> &uu, Field<double> &kx, Field<double> &ja, Dom &dom, Dom &inner, Dom &uc_mapper, Dom &uu_mapper, int mpi_size, int mpi_rank, MPI_Request *req, double &kernel_time1, double &kernel_time2, double &comm_time) {
    dim3 &isz =     inner._h._size;
    dim3 &osz =       dom._h._size;
    dim3 &csz = uc_mapper._h._size;
    dim3 &usz = uu_mapper._h._size;

    unsigned int idx_start, idx_end;
    idx_start = (mpi_rank >  0           )? FALMUtil::d321(    0,0,0,csz) : FALMUtil::d321(      1,0,0,csz);
    idx_end   = (mpi_rank == mpi_size - 1)? FALMUtil::d321(csz.x,0,0,csz) : FALMUtil::d321(csz.x+1,0,0,csz);
    interp_psvelocity_kernel_1<<<n_blocks, n_threads>>>(*(u._dd), *(uc._dd), *(kx._dd), *(ja._dd), *(dom._d), *(uc_mapper._d), idx_start, idx_end);

    if (mpi_size > 1) {
        
    }
}

}

#endif