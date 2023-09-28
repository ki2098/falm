#include "../structEqL1.h"
#include "../MVL1.h"
#include "devutil.cuh"

namespace Falm {

__global__ void kernel_Struct3d7p_MV(MatrixFrame<REAL> &a, MatrixFrame<REAL> &x, MatrixFrame<REAL> &ax, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxc = IDX(i  , j  , k  , pdom_shape);
        INT idxe = IDX(i+1, j  , k  , pdom_shape);
        INT idxw = IDX(i-1, j  , k  , pdom_shape);
        INT idxn = IDX(i  , j+1, k  , pdom_shape);
        INT idxs = IDX(i  , j-1, k  , pdom_shape);
        INT idxt = IDX(i  , j  , k+1, pdom_shape);
        INT idxb = IDX(i  , j  , k-1, pdom_shape);
        REAL ac = a(idxc, 0);
        REAL ae = a(idxc, 1);
        REAL aw = a(idxc, 2);
        REAL an = a(idxc, 3);
        REAL as = a(idxc, 4);
        REAL at = a(idxc, 5);
        REAL ab = a(idxc, 6);
        REAL xc = x(idxc);
        REAL xe = x(idxe);
        REAL xw = x(idxw);
        REAL xn = x(idxn);
        REAL xs = x(idxs);
        REAL xt = x(idxt);
        REAL xb = x(idxb);
        ax(idxc) = ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb;
    }
}

void L0Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdom, Mapper &map, dim3 block_dim) {
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

__global__ void kernel_Struct3d7p_Res(MatrixFrame<REAL> &a, MatrixFrame<REAL> &x, MatrixFrame<REAL> &b, MatrixFrame<REAL> &r, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxc = IDX(i  , j  , k  , pdom_shape);
        INT idxe = IDX(i+1, j  , k  , pdom_shape);
        INT idxw = IDX(i-1, j  , k  , pdom_shape);
        INT idxn = IDX(i  , j+1, k  , pdom_shape);
        INT idxs = IDX(i  , j-1, k  , pdom_shape);
        INT idxt = IDX(i  , j  , k+1, pdom_shape);
        INT idxb = IDX(i  , j  , k-1, pdom_shape);
        REAL ac = a(idxc, 0);
        REAL ae = a(idxc, 1);
        REAL aw = a(idxc, 2);
        REAL an = a(idxc, 3);
        REAL as = a(idxc, 4);
        REAL at = a(idxc, 5);
        REAL ab = a(idxc, 6);
        REAL xc = x(idxc);
        REAL xe = x(idxe);
        REAL xw = x(idxw);
        REAL xn = x(idxn);
        REAL xs = x(idxs);
        REAL xt = x(idxt);
        REAL xb = x(idxb);
        r(idxc) = b(idxc) - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb);
    }
}

void L0Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdom, Mapper &map, dim3 block_dim) {
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

__global__ void kernel_Struct3d7p_Jacobi(MatrixFrame<REAL> &a, MatrixFrame<REAL> &x, MatrixFrame<REAL> &xp, MatrixFrame<REAL> &b, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxc = IDX(i  , j  , k  , pdom_shape);
        INT idxe = IDX(i+1, j  , k  , pdom_shape);
        INT idxw = IDX(i-1, j  , k  , pdom_shape);
        INT idxn = IDX(i  , j+1, k  , pdom_shape);
        INT idxs = IDX(i  , j-1, k  , pdom_shape);
        INT idxt = IDX(i  , j  , k+1, pdom_shape);
        INT idxb = IDX(i  , j  , k-1, pdom_shape);
        REAL ac =  a(idxc, 0);
        REAL ae =  a(idxc, 1);
        REAL aw =  a(idxc, 2);
        REAL an =  a(idxc, 3);
        REAL as =  a(idxc, 4);
        REAL at =  a(idxc, 5);
        REAL ab =  a(idxc, 6);
        REAL xc = xp(idxc);
        REAL xe = xp(idxe);
        REAL xw = xp(idxw);
        REAL xn = xp(idxn);
        REAL xs = xp(idxs);
        REAL xt = xp(idxt);
        REAL xb = xp(idxb);
        x(idxc) = xc + (b(idxc) - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb)) / ac;
    }
}

void L1EqSolver::L0Dev_Struct3d7p_JacobiSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &xp, Matrix<REAL> &b, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_Jacobi<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(xp.devptr), *(b.devptr), pdom.shape, map.shape, map.offset);
}

void L1EqSolver::L1Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, "Jacobi" + x.name + "Previous");
    it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, map, block_dim);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(L0Dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L1EqSolver::L1Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, "Jacobi" + x.name + "Previous");
    INT __it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, map, block_dim);
        __it ++;
    } while (__it < pc_maxit);
}

__global__ void kernel_Struct3d7p_SOR(MatrixFrame<REAL> &a, MatrixFrame<REAL> &x, MatrixFrame<REAL> &b, REAL omega, INT color, INTx3 pdom_shape, INTx3 pdom_offset, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idxc = IDX(i  , j  , k  , pdom_shape);
        INT idxe = IDX(i+1, j  , k  , pdom_shape);
        INT idxw = IDX(i-1, j  , k  , pdom_shape);
        INT idxn = IDX(i  , j+1, k  , pdom_shape);
        INT idxs = IDX(i  , j-1, k  , pdom_shape);
        INT idxt = IDX(i  , j  , k+1, pdom_shape);
        INT idxb = IDX(i  , j  , k-1, pdom_shape);
        REAL ac = a(idxc, 0);
        REAL ae = a(idxc, 1);
        REAL aw = a(idxc, 2);
        REAL an = a(idxc, 3);
        REAL as = a(idxc, 4);
        REAL at = a(idxc, 5);
        REAL ab = a(idxc, 6);
        REAL xc = x(idxc);
        REAL xe = x(idxe);
        REAL xw = x(idxw);
        REAL xn = x(idxn);
        REAL xs = x(idxs);
        REAL xt = x(idxt);
        REAL xb = x(idxb);
        REAL bc = b(idxc);
        REAL cc = 0;
        if ((i + j + k + SUM3(pdom_offset)) % 2 == color) {
            cc = (bc - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb)) / ac;
        }
        x(idxc) = xc + omega * cc;
    }
}

void L1EqSolver::L0Dev_Struct3d7p_SORSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, INT color, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_SOR<<<grid_dim, block_dim, 0, 0>>>(*(a.devptr), *(x.devptr), *(b.devptr), omega, color, pdom.shape, pdom.offset, map.shape, map.offset);
}

void L1EqSolver::L1Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );

    it = 0;
    do {
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdom, map, block_dim);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red  , pdom, map, block_dim);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
        err = sqrt(L0Dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L1EqSolver::L1Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    INT __it = 0;
    do {
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdom, map, block_dim);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red  , pdom, map, block_dim);
         __it ++;
    } while (__it < pc_maxit);
}

__global__ void kernel_PBiCGStab_1(MatrixFrame<REAL> &p, MatrixFrame<REAL> &q, MatrixFrame<REAL> &r, REAL beta, REAL omega, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    }
}

void L1EqSolver::L0Dev_PBiCGStab1(Matrix<REAL> &p, Matrix<REAL> &q, Matrix<REAL> &r, REAL beta, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_1<<<grid_dim, block_dim, 0, 0>>>(*(p.devptr), *(q.devptr), *(r.devptr), beta, omega, pdom.shape, map.shape, map.offset);
}

__global__ void kernel_PBiCGStab_2(MatrixFrame<REAL> &s, MatrixFrame<REAL> &q, MatrixFrame<REAL> &r, REAL alpha, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        s(idx) = r(idx) - alpha * q(idx);
    }
}

void L1EqSolver::L0Dev_PBiCGStab2(Matrix<REAL> &s, Matrix<REAL> &q, Matrix<REAL> &r, REAL alpha, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_2<<<grid_dim, block_dim, 0, 0>>>(*(s.devptr), *(q.devptr), *(r.devptr), alpha, pdom.shape, map.shape, map.offset);
}

__global__ void kernel_PBiCGStab_3(MatrixFrame<REAL> &x, MatrixFrame<REAL> &pp, MatrixFrame<REAL> &ss, REAL alpha, REAL omega, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        x(idx) += alpha * pp(idx) + omega * ss(idx);
    }
}

void L1EqSolver::L0Dev_PBiCGStab3(Matrix<REAL> &x, Matrix<REAL> &pp, Matrix<REAL> &ss, REAL alpha, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_3<<<grid_dim, block_dim, 0, 0>>>(*(x.devptr), *(pp.devptr), *(ss.devptr), alpha, omega, pdom.shape, map.shape, map.offset);
} 

__global__ void kernel_PBiCGStab_4(MatrixFrame<REAL> &r, MatrixFrame<REAL> &s, MatrixFrame<REAL> &t, REAL omega, INTx3 pdom_shape, INTx3 map_shape, INTx3 map_offset) {
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape.x && j < map_shape.y && k < map_shape.z) {
        i += map_offset.x;
        j += map_offset.y;
        k += map_offset.z;
        INT idx = IDX(i, j, k, pdom_shape);
        r(idx) = s(idx) - omega * t(idx);
    }
}

void L1EqSolver::L0Dev_PBiCGStab4(Matrix<REAL> &r, Matrix<REAL> &s, Matrix<REAL> &t, REAL omega, Mapper &pdom, Mapper &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_4<<<grid_dim, block_dim, 0, 0>>>(*(r.devptr), *(s.devptr), *(t.devptr), omega, pdom.shape, map.shape, map.offset);
}

void L1EqSolver::L1Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim) {
    Mapper map(pdom, Gd);
    assert(
        a.shape.x == x.shape.x && a.shape.x == b.shape.x && a.shape.x == r.shape.x &&
        a.shape.y == 7 && x.shape.y == 1 && b.shape.y == 1 && r.shape.y == 1
    );
    dim3 grid_dim(
        (map.shape.x + block_dim.x - 1) / block_dim.x,
        (map.shape.y + block_dim.y - 1) / block_dim.y,
        (map.shape.z + block_dim.z - 1) / block_dim.z
    );

    Matrix<REAL> rr(pdom.shape, 1, HDCType::Device, "PBiCGStab rr");
    Matrix<REAL>  p(pdom.shape, 1, HDCType::Device, "PBiCGStab  p");
    Matrix<REAL>  q(pdom.shape, 1, HDCType::Device, "PBiCGStab  q");
    Matrix<REAL>  s(pdom.shape, 1, HDCType::Device, "PBiCGStab  s");
    Matrix<REAL> pp(pdom.shape, 1, HDCType::Device, "PBiCGStab pp");
    Matrix<REAL> ss(pdom.shape, 1, HDCType::Device, "PBiCGStab ss");
    Matrix<REAL>  t(pdom.shape, 1, HDCType::Device, "PBiCGStab  t");

    REAL rho, rrho, alpha, beta, omega;

    L0Dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim);
    err = sqrt(L0Dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;
    rr.cpy(r, HDCType::Device);

    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        if (err < tol) {
            break;
        }

        rho = L0Dev_DotProduct(r, rr, pdom, map, block_dim);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            // kernel_PBiCGStab_1<<<grid_dim, block_dim>>>(*(p.devptr), *(q.devptr), *(r.devptr), beta, omega, pdom.shape, map.shape, map.offset);
            L0Dev_PBiCGStab1(p, q, r, beta, omega, pdom, map, block_dim);
        }
        pp.clear(HDCType::Device);
        L1Dev_Struct3d7p_Precondition(a, pp, p, pdom, block_dim);
        L0Dev_Struct3d7p_MV(a, pp, q, pdom, map, block_dim);
        alpha = rho / L0Dev_DotProduct(rr, q, pdom, map, block_dim);

        // kernel_PBiCGStab_2<<<grid_dim, block_dim>>>(*(s.devptr), *(q.devptr), *(r.devptr), alpha, pdom.shape, map.shape, map.offset);
        L0Dev_PBiCGStab2(s, q, r, alpha, pdom, map, block_dim);
        ss.clear(HDCType::Device);
        L1Dev_Struct3d7p_Precondition(a, ss, s, pdom, block_dim);
        L0Dev_Struct3d7p_MV(a, ss, t, pdom, map, block_dim);
        omega = L0Dev_DotProduct(t, s, pdom, map, block_dim) / L0Dev_DotProduct(t, t, pdom, map, block_dim);

        // kernel_PBiCGStab_3<<<grid_dim, block_dim, 0, 0>>>(*(x.devptr), *(pp.devptr), *(ss.devptr), alpha, omega, pdom.shape, map.shape, map.offset);
        // kernel_PBiCGStab_4<<<grid_dim, block_dim, 0, 0>>>(*(r.devptr), *(s.devptr), *(t.devptr), omega, pdom.shape, map.shape, map.offset);
        L0Dev_PBiCGStab3(x, pp, ss, alpha, omega, pdom, map, block_dim);
        L0Dev_PBiCGStab4(r, s, t, omega, pdom, map, block_dim);

        rrho = rho;

        err = sqrt(L0Dev_Norm2Sq(r, pdom, map, block_dim)) / map.size;
        it ++;
    } while (it < maxit && err > tol);
}

}