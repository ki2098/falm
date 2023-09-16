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

void StructLEqSolver::dev_Struct3d7p_JacobiSweep(Matrix<double> &a, Matrix<double> &x, Matrix<double> &xp, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    xp.cpy(x, HDCType::Device);
    kernel_Struct3d7p_Jacobi<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(xp.devptr), *(b.devptr), pdom.shape, map.shape, map.offset);
}

void StructLEqSolver::dev_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );

    Matrix<double> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    it = 0;
    do {
        dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, map, block_dim);
        dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim));
        it ++;
    } while (it < maxit && err > tol);
}

void StructLEqSolver::dev_Struct3d7p_JAcobiPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    Matrix<double> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    int __it = 0;
    do {
        dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, map, block_dim);
        __it ++;
    } while (__it < pc_maxit);
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

void StructLEqSolver::dev_Struct3d7p_SORSweep(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, double omega, unsigned int color, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_SOR<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(b.devptr), omega, color, pdom.shape, pdom.offset, map.shape, map.offset);
}

void StructLEqSolver::dev_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );

    it = 0;
    do {
        dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdom, map, block_dim);
        dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red  , pdom, map, block_dim);
        dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim));
        it ++;
    } while (it < maxit && err > tol);
}

void StructLEqSolver::dev_Struct3d7p_SORPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    int __it = 0;
    do {
        dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdom, map, block_dim);
        dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red  , pdom, map, block_dim);
         __it ++;
    } while (__it < pc_maxit);
}

__global__ void kernel_PBiCGStab_1(MatrixFrame<double> &p, MatrixFrame<double> &q, MatrixFrame<double> &r, double beta, double omega, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    }
}

__global__ void kernel_PBiCGStab_2(MatrixFrame<double> &s, MatrixFrame<double> &q, MatrixFrame<double> &r, double alpha, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        s(idx) = r(idx) - alpha * q(idx);
    }
}

__global__ void kernel_PBiCGStab_3(MatrixFrame<double> &x, MatrixFrame<double> &pp, MatrixFrame<double> &ss, double alpha, double omega, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        x(idx) += alpha * pp(idx) + omega * ss(idx);
    }
}

__global__ void kernel_PBiCGStab_4(MatrixFrame<double> &r, MatrixFrame<double> &s, MatrixFrame<double> &t, double omega, uint3 pdom_shape, uint3 map_shape, uint3 map_offset) {
    unsigned int i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        unsigned int idx = IDX(i, j, k, pdom_shape);
        r(idx) = s(idx) - omega * t(idx);
    }
}

void StructLEqSolver::dev_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, Mapper &map, dim3 &block_dim) {
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    Matrix<double> rr(pdom.shape, 1, HDCType::Device, 101);
    Matrix<double>  p(pdom.shape, 1, HDCType::Device, 102);
    Matrix<double>  q(pdom.shape, 1, HDCType::Device, 103);
    Matrix<double>  s(pdom.shape, 1, HDCType::Device, 104);
    Matrix<double> pp(pdom.shape, 1, HDCType::Device, 105);
    Matrix<double> ss(pdom.shape, 1, HDCType::Device, 106);
    Matrix<double>  t(pdom.shape, 1, HDCType::Device, 107);

    double rho, rrho, alpha, beta, omega;

    dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
    err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;
    rr.cpy(r, HDCType::Device);

    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        if (err < tol) {
            break;
        }

        rho = dev_DotProduct(r, rr, pdom, map, block_dim);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            kernel_PBiCGStab_1<<<grid_dim, block_dim>>>(*(p.devptr), *(q.devptr), *(r.devptr), beta, omega, pdom.shape, map.shape, map.offset);
        }
        pp.clear(HDCType::Device);
        dev_Struct3d7p_Precondition(a, pp, p, pdom, map, block_dim);
        dev_Struct3d7p_MV(a, pp, q, pdom, map, block_dim);
        alpha = rho / dev_DotProduct(rr, q, pdom, map, block_dim);

        kernel_PBiCGStab_2<<<grid_dim, block_dim>>>(*(s.devptr), *(q.devptr), *(r.devptr), alpha, pdom.shape, map.shape, map.offset);
        ss.clear(HDCType::Device);
        dev_Struct3d7p_Precondition(a, ss, s, pdom, map, block_dim);
        dev_Struct3d7p_MV(a, ss, t, pdom, map, block_dim);
        omega = dev_DotProduct(t, s, pdom, map, block_dim) / dev_DotProduct(t, t, pdom, map, block_dim);

        kernel_PBiCGStab_3<<<grid_dim, block_dim, 0, 0>>>(*(x.devptr), *(pp.devptr), *(ss.devptr), alpha, omega, pdom.shape, map.shape, map.offset);
        kernel_PBiCGStab_4<<<grid_dim, block_dim, 0, 0>>>(*(r.devptr), *(s.devptr), *(t.devptr), omega, pdom.shape, map.shape, map.offset);

        rrho = rho;

        err = sqrt(dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;

        it ++;
    } while (it < maxit && err > tol);
}

}