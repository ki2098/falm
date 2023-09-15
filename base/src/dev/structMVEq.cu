#include "../structMVEq.h"
#include "../mvbasic.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_Struct3d7p_MV(MatrixFrame<double> &a, MatrixFrame<double> &x, MatrixFrame<double> &ax, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idxc = IDX(i  , j  , k  , pdom_shape);
        unsigned int idxe = IDX(i+1, j  , k  , pdom_shape);
        unsigned int idxw = IDX(i-1, j  , k  , pdom_shape);
        unsigned int idxn = IDX(i  , j+1, k  , pdom_shape);
        unsigned int idxs = IDX(i  , j-1, k  , pdom_shape);
        unsigned int idxt = IDX(i  , j  , k+1, pdom_shape);
        unsigned int idxb = IDX(i  , j  , k-1, pdom_shape);
        double ac = a(idxc, 0);
        double ae = a(idxc, 1);
        double aw = a(idxc, 2);
        double an = a(idxc, 3);
        double as = a(idxc, 4);
        double at = a(idxc, 5);
        double ab = a(idxc, 6);
        double xc = x(idxc);
        double xe = x(idxe);
        double xw = x(idxw);
        double xn = x(idxn);
        double xs = x(idxs);
        double xt = x(idxt);
        double xb = x(idxb);
        ax(idxc) = ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb;
    }
}

void dev_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == ax.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && ax.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_MV<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(ax.devptr), pdom.shape, map.shape, map.offset);
}

__global__ void kernel_Struct3d7p_Res(MatrixFrame<double> &a, MatrixFrame<double> &x, MatrixFrame<double> &b, MatrixFrame<double> &r, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idxc = IDX(i  , j  , k  , pdom_shape);
        unsigned int idxe = IDX(i+1, j  , k  , pdom_shape);
        unsigned int idxw = IDX(i-1, j  , k  , pdom_shape);
        unsigned int idxn = IDX(i  , j+1, k  , pdom_shape);
        unsigned int idxs = IDX(i  , j-1, k  , pdom_shape);
        unsigned int idxt = IDX(i  , j  , k+1, pdom_shape);
        unsigned int idxb = IDX(i  , j  , k-1, pdom_shape);
        double ac = a(idxc, 0);
        double ae = a(idxc, 1);
        double aw = a(idxc, 2);
        double an = a(idxc, 3);
        double as = a(idxc, 4);
        double at = a(idxc, 5);
        double ab = a(idxc, 6);
        double xc = x(idxc);
        double xe = x(idxe);
        double xw = x(idxw);
        double xn = x(idxn);
        double xs = x(idxs);
        double xt = x(idxt);
        double xb = x(idxb);
        r(idxc) = b(idxc) - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb);
    }
}

void dev_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_Res<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(b.devptr), *(r.devptr), pdom.shape, map.shape, map.offset);
}

__global__ void kernel_Struct3d7p_Jacobi(MatrixFrame<double> &a, MatrixFrame<double> &x, MatrixFrame<double> &xp, MatrixFrame<double> &b, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idxc = IDX(i  , j  , k  , pdom_shape);
        unsigned int idxe = IDX(i+1, j  , k  , pdom_shape);
        unsigned int idxw = IDX(i-1, j  , k  , pdom_shape);
        unsigned int idxn = IDX(i  , j+1, k  , pdom_shape);
        unsigned int idxs = IDX(i  , j-1, k  , pdom_shape);
        unsigned int idxt = IDX(i  , j  , k+1, pdom_shape);
        unsigned int idxb = IDX(i  , j  , k-1, pdom_shape);
        double ac =  a(idxc, 0);
        double ae =  a(idxc, 1);
        double aw =  a(idxc, 2);
        double an =  a(idxc, 3);
        double as =  a(idxc, 4);
        double at =  a(idxc, 5);
        double ab =  a(idxc, 6);
        double xc = xp(idxc);
        double xe = xp(idxe);
        double xw = xp(idxw);
        double xn = xp(idxn);
        double xs = xp(idxs);
        double xt = xp(idxt);
        double xb = xp(idxb);
        x(idxc) = xc + (b(idxc) - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb)) / ac;
    }
}

__global__ void kernel_Struct3d7p_SOR(MatrixFrame<double> &a, MatrixFrame<double> &x, MatrixFrame<double> &b, double omega, unsigned int color, uint3 pdom_shape, uint3 pdom_offset, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idxc = IDX(i  , j  , k  , pdom_shape);
        unsigned int idxe = IDX(i+1, j  , k  , pdom_shape);
        unsigned int idxw = IDX(i-1, j  , k  , pdom_shape);
        unsigned int idxn = IDX(i  , j+1, k  , pdom_shape);
        unsigned int idxs = IDX(i  , j-1, k  , pdom_shape);
        unsigned int idxt = IDX(i  , j  , k+1, pdom_shape);
        unsigned int idxb = IDX(i  , j  , k-1, pdom_shape);
        double ac = a(idxc, 0);
        double ae = a(idxc, 1);
        double aw = a(idxc, 2);
        double an = a(idxc, 3);
        double as = a(idxc, 4);
        double at = a(idxc, 5);
        double ab = a(idxc, 6);
        double xc = x(idxc);
        double xe = x(idxe);
        double xw = x(idxw);
        double xn = x(idxn);
        double xs = x(idxs);
        double xt = x(idxt);
        double xb = x(idxb);
        double bc = b(idxc);
        double cc = 0;
        if ((i + j + k + SUM3(pdom_offset)) % 2 == color) {
            cc = (bc - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb)) / ac;
        }
        x(idxc) = xc + omega * cc;
    }
}

void StructLEqSolver::dev_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    Matrix<double> xp(x.shape.x, x.shape.y, HDCTYPE::Device, x.label);
    it = 0;
    do {
        xp.cpy(x, HDCTYPE::Device);
        kernel_Struct3d7p_Jacobi<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(xp.devptr), *(b.devptr), pdom.shape, map.shape, map.offset);
        dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim));
        it ++;
    } while (it < maxit && err > tol);
}

void StructLEqSolver::dev_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    it = 0;
    do {
        kernel_Struct3d7p_SOR<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(b.devptr), relax_factor, COLOR::Black, pdom.shape, pdom.offset, map.shape, map.offset);
        kernel_Struct3d7p_SOR<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(b.devptr), relax_factor, COLOR::Red  , pdom.shape, pdom.offset, map.shape, map.offset);
        dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim));
        it ++;
    } while (it < maxit && err > tol);
}

}