#include "../FalmEqDevCall.h"
#include "../MVDevCall.h"
#include "devutil.cuh"

namespace Falm {

// __global__ void kernel_Struct3d7p_MV(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vax, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
//     const MatrixFrame<REAL> &a=*va, &x=*vx, &ax=*vax;
//     INT i, j, k;
//     GLOBAL_THREAD_IDX_3D(i, j, k);
//     if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
//         i += map_offset[0];
//         j += map_offset[1];
//         k += map_offset[2];
//         INT idxc = IDX(i  , j  , k  , pdm_shape);
//         INT idxe = IDX(i+1, j  , k  , pdm_shape);
//         INT idxw = IDX(i-1, j  , k  , pdm_shape);
//         INT idxn = IDX(i  , j+1, k  , pdm_shape);
//         INT idxs = IDX(i  , j-1, k  , pdm_shape);
//         INT idxt = IDX(i  , j  , k+1, pdm_shape);
//         INT idxb = IDX(i  , j  , k-1, pdm_shape);
//         REAL ac = a(idxc, 0);
//         REAL ae = a(idxc, 1);
//         REAL aw = a(idxc, 2);
//         REAL an = a(idxc, 3);
//         REAL as = a(idxc, 4);
//         REAL at = a(idxc, 5);
//         REAL ab = a(idxc, 6);
//         REAL xc = x(idxc);
//         REAL xe = x(idxe);
//         REAL xw = x(idxw);
//         REAL xn = x(idxn);
//         REAL xs = x(idxs);
//         REAL xt = x(idxt);
//         REAL xb = x(idxb);
//         ax(idxc) = ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb;
//     }
// }

// void MVDevCall::MVMult(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
//     assert(
//         a.shape[0] == x.shape[0] && a.shape[0] == ax.shape[0] &&
//         a.shape[1] == 7 && x.shape[1] == 1 && ax.shape[1] == 1
//     );
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );

//     kernel_Struct3d7p_MV<<<grid_dim, block_dim, 0, stream>>>(a.devptr, x.devptr, ax.devptr, pdm.shape, map.shape, map.offset);
// }

__global__ void kernel_Struct3d7p_Res(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vb, const MatrixFrame<REAL> *vr, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va, &x=*vx, &b=*vb, &r=*vr;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i  , j  , k  , pdm_shape);
        INT idxe = IDX(i+1, j  , k  , pdm_shape);
        INT idxw = IDX(i-1, j  , k  , pdm_shape);
        INT idxn = IDX(i  , j+1, k  , pdm_shape);
        INT idxs = IDX(i  , j-1, k  , pdm_shape);
        INT idxt = IDX(i  , j  , k+1, pdm_shape);
        INT idxb = IDX(i  , j  , k-1, pdm_shape);
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

void FalmEqDevCall::Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
    assert(
        a.shape[0] == x.shape[0] && a.shape[0] == b.shape[0] && a.shape[0] == r.shape[0] &&
        a.shape[1] == 7 && x.shape[1] == 1 && b.shape[1] == 1 && r.shape[1] == 1
    );
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_Res<<<grid_dim, block_dim, 0, stream>>>(a.devptr, x.devptr, b.devptr, r.devptr, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_Struct3d7p_Jacobi(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vxp, const MatrixFrame<REAL> *vb, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va, &x=*vx, &xp=*vxp, &b=*vb;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i  , j  , k  , pdm_shape);
        INT idxe = IDX(i+1, j  , k  , pdm_shape);
        INT idxw = IDX(i-1, j  , k  , pdm_shape);
        INT idxn = IDX(i  , j+1, k  , pdm_shape);
        INT idxs = IDX(i  , j-1, k  , pdm_shape);
        INT idxt = IDX(i  , j  , k+1, pdm_shape);
        INT idxb = IDX(i  , j  , k-1, pdm_shape);
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

void FalmEqDevCall::JacobiSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &xp, Matrix<REAL> &b, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_Jacobi<<<grid_dim, block_dim, 0, stream>>>(a.devptr, x.devptr, xp.devptr, b.devptr, pdm.shape, map.shape, map.offset);
}

// void FalmEqDevCall::L1Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     assert(
//         a.shape[0] == x.shape[0] && a.shape[0] == b.shape[0] && a.shape[0] == r.shape[0] &&
//         a.shape[1] == 7 && x.shape[1] == 1 && b.shape[1] == 1 && r.shape[1] == 1
//     );

//     Matrix<REAL> xp(x.shape[0], x.shape[1], HDCType::Device, "Jacobi" + x.name + "Previous");
//     it = 0;
//     do {
//         xp.cpy(x, HDCType::Device);
//         JacobiSweep(a, x, xp, b, pdm, map, block_dim);
//         Res(a, x, b, r, pdm, map, block_dim);
//         err = sqrt(MVDevCall::EuclideanNormSq(r, pdm, map, block_dim)) / map.size;
//         it ++;
//     } while (it < maxit && err > tol);
// }

// void FalmEqDevCall::L1Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     Matrix<REAL> xp(x.shape[0], x.shape[1], HDCType::Device, "Jacobi" + x.name + "Previous");
//     INT __it = 0;
//     do {
//         xp.cpy(x, HDCType::Device);
//         JacobiSweep(a, x, xp, b, pdm, map, block_dim);
//         __it ++;
//     } while (__it < pc_maxit);
// }

__global__ void kernel_Struct3d7p_SOR(const MatrixFrame<REAL> *va, const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vb, REAL omega, INT color, INT3 pdm_shape, INT3 pdm_offset, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &a=*va, &x=*vx, &b=*vb;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idxc = IDX(i  , j  , k  , pdm_shape);
        INT idxe = IDX(i+1, j  , k  , pdm_shape);
        INT idxw = IDX(i-1, j  , k  , pdm_shape);
        INT idxn = IDX(i  , j+1, k  , pdm_shape);
        INT idxs = IDX(i  , j-1, k  , pdm_shape);
        INT idxt = IDX(i  , j  , k+1, pdm_shape);
        INT idxb = IDX(i  , j  , k-1, pdm_shape);
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
        if ((i + j + k + SUM3(pdm_offset)) % 2 == color) {
            cc = (bc - (ac * xc + ae * xe + aw * xw + an * xn + as * xs + at * xt + ab * xb)) / ac;
        }
        x(idxc) = xc + omega * cc;
    }
}

void FalmEqDevCall::SORSweep(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, REAL omega, INT color, Region &pdm, const Region &map, dim3 block_dim, STREAM stream) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );

    kernel_Struct3d7p_SOR<<<grid_dim, block_dim, 0, stream>>>(a.devptr, x.devptr, b.devptr, omega, color, pdm.shape, pdm.offset, map.shape, map.offset);
}

// void FalmEqDevCall::L1Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     assert(
//         a.shape[0] == x.shape[0] && a.shape[0] == b.shape[0] && a.shape[0] == r.shape[0] &&
//         a.shape[1] == 7 && x.shape[1] == 1 && b.shape[1] == 1 && r.shape[1] == 1
//     );

//     it = 0;
//     do {
//         SORSweep(a, x, b, relax_factor, Color::Black, pdm, map, block_dim);
//         SORSweep(a, x, b, relax_factor, Color::Red  , pdm, map, block_dim);
//         Res(a, x, b, r, pdm, map, block_dim);
//         err = sqrt(MVDevCall::EuclideanNormSq(r, pdm, map, block_dim)) / map.size;
//         it ++;
//     } while (it < maxit && err > tol);
// }

// void FalmEqDevCall::L1Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     INT __it = 0;
//     do {
//         SORSweep(a, x, b, pc_relax_factor, Color::Black, pdm, map, block_dim);
//         SORSweep(a, x, b, pc_relax_factor, Color::Red  , pdm, map, block_dim);
//          __it ++;
//     } while (__it < pc_maxit);
// }

__global__ void kernel_PBiCGStab_1(const MatrixFrame<REAL> *vp, const MatrixFrame<REAL> *vq, const MatrixFrame<REAL> *vr, REAL beta, REAL omega, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &p=*vp, &q=*vq, &r=*vr;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        p(idx) = r(idx) + beta * (p(idx) - omega * q(idx));
    }
}

void FalmEqDevCall::PBiCGStab1(Matrix<REAL> &p, Matrix<REAL> &q, Matrix<REAL> &r, REAL beta, REAL omega, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_1<<<grid_dim, block_dim, 0, 0>>>(p.devptr, q.devptr, r.devptr, beta, omega, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_PBiCGStab_2(const MatrixFrame<REAL> *vs, const MatrixFrame<REAL> *vq, const MatrixFrame<REAL> *vr, REAL alpha, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &s=*vs, &q=*vq, &r=*vr;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        s(idx) = r(idx) - alpha * q(idx);
    }
}

void FalmEqDevCall::PBiCGStab2(Matrix<REAL> &s, Matrix<REAL> &q, Matrix<REAL> &r, REAL alpha, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_2<<<grid_dim, block_dim, 0, 0>>>(s.devptr, q.devptr, r.devptr, alpha, pdm.shape, map.shape, map.offset);
}

__global__ void kernel_PBiCGStab_3(const MatrixFrame<REAL> *vx, const MatrixFrame<REAL> *vpp, const MatrixFrame<REAL> *vss, REAL alpha, REAL omega, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &x=*vx, &pp=*vpp, &ss=*vss;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        x(idx) += alpha * pp(idx) + omega * ss(idx);
    }
}

void FalmEqDevCall::PBiCGStab3(Matrix<REAL> &x, Matrix<REAL> &pp, Matrix<REAL> &ss, REAL alpha, REAL omega, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_3<<<grid_dim, block_dim, 0, 0>>>(x.devptr, pp.devptr, ss.devptr, alpha, omega, pdm.shape, map.shape, map.offset);
} 

__global__ void kernel_PBiCGStab_4(const MatrixFrame<REAL> *vr, const MatrixFrame<REAL> *vs, const MatrixFrame<REAL> *vt, REAL omega, INT3 pdm_shape, INT3 map_shape, INT3 map_offset) {
    const MatrixFrame<REAL> &r=*vr, &s=*vs, &t=*vt;
    INT i, j, k;
    GLOBAL_THREAD_IDX_3D(i, j, k);
    if (i < map_shape[0] && j < map_shape[1] && k < map_shape[2]) {
        i += map_offset[0];
        j += map_offset[1];
        k += map_offset[2];
        INT idx = IDX(i, j, k, pdm_shape);
        r(idx) = s(idx) - omega * t(idx);
    }
}

void FalmEqDevCall::PBiCGStab4(Matrix<REAL> &r, Matrix<REAL> &s, Matrix<REAL> &t, REAL omega, Region &pdm, const Region &map, dim3 block_dim) {
    dim3 grid_dim(
        (map.shape[0] + block_dim.x - 1) / block_dim.x,
        (map.shape[1] + block_dim.y - 1) / block_dim.y,
        (map.shape[2] + block_dim.z - 1) / block_dim.z
    );
    kernel_PBiCGStab_4<<<grid_dim, block_dim, 0, 0>>>(r.devptr, s.devptr, t.devptr, omega, pdm.shape, map.shape, map.offset);
}

// void FalmEqDevCall::L1Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPMBase &cpm, dim3 block_dim) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     Region map(pdm.shape, cpm.gc);
//     assert(
//         a.shape[0] == x.shape[0] && a.shape[0] == b.shape[0] && a.shape[0] == r.shape[0] &&
//         a.shape[1] == 7 && x.shape[1] == 1 && b.shape[1] == 1 && r.shape[1] == 1
//     );
//     dim3 grid_dim(
//         (map.shape[0] + block_dim.x - 1) / block_dim.x,
//         (map.shape[1] + block_dim.y - 1) / block_dim.y,
//         (map.shape[2] + block_dim.z - 1) / block_dim.z
//     );

//     Matrix<REAL> rr(pdm.shape, 1, HDCType::Device, "PBiCGStab rr");
//     Matrix<REAL>  p(pdm.shape, 1, HDCType::Device, "PBiCGStab  p");
//     Matrix<REAL>  q(pdm.shape, 1, HDCType::Device, "PBiCGStab  q");
//     Matrix<REAL>  s(pdm.shape, 1, HDCType::Device, "PBiCGStab  s");
//     Matrix<REAL> pp(pdm.shape, 1, HDCType::Device, "PBiCGStab pp");
//     Matrix<REAL> ss(pdm.shape, 1, HDCType::Device, "PBiCGStab ss");
//     Matrix<REAL>  t(pdm.shape, 1, HDCType::Device, "PBiCGStab  t");

//     REAL rho, rrho, alpha, beta, omega;

//     Res(a, x, b, r, pdm, map, block_dim);
//     err = sqrt(MVDevCall::EuclideanNormSq(r, pdm, map, block_dim)) / map.size;
//     rr.cpy(r, HDCType::Device);

//     rrho  = 1.0;
//     alpha = 0.0;
//     omega = 1.0;

//     it = 0;
//     do {
//         // if (err < tol) {
//         //     break;
//         // }

//         rho = MVDevCall::DotProduct(r, rr, pdm, map, block_dim);
//         if (fabs(rho) < __FLT_MIN__) {
//             err = rho;
//             break;
//         }

//         if (it == 0) {
//             p.cpy(r, HDCType::Device);
//         } else {
//             beta = (rho * alpha) / (rrho * omega);
//             // kernel_PBiCGStab_1<<<grid_dim, block_dim>>>(p.devptr, q.devptr, r.devptr, beta, omega, pdm.shape, map.shape, map.offset);
//             PBiCGStab1(p, q, r, beta, omega, pdm, map, block_dim);
//         }
//         pp.clear(HDCType::Device);
//         L1Dev_Struct3d7p_Precondition(a, pp, p, cpm, block_dim);
//         MVDevCall::MVMult(a, pp, q, pdm, map, block_dim);
//         alpha = rho / MVDevCall::DotProduct(rr, q, pdm, map, block_dim);

//         // kernel_PBiCGStab_2<<<grid_dim, block_dim>>>(s.devptr, q.devptr, r.devptr, alpha, pdm.shape, map.shape, map.offset);
//         PBiCGStab2(s, q, r, alpha, pdm, map, block_dim);
//         ss.clear(HDCType::Device);
//         L1Dev_Struct3d7p_Precondition(a, ss, s, cpm, block_dim);
//         MVDevCall::MVMult(a, ss, t, pdm, map, block_dim);
//         omega = MVDevCall::DotProduct(t, s, pdm, map, block_dim) / MVDevCall::DotProduct(t, t, pdm, map, block_dim);

//         // kernel_PBiCGStab_3<<<grid_dim, block_dim, 0, 0>>>(x.devptr, pp.devptr, ss.devptr, alpha, omega, pdm.shape, map.shape, map.offset);
//         // kernel_PBiCGStab_4<<<grid_dim, block_dim, 0, 0>>>(r.devptr, s.devptr, t.devptr, omega, pdm.shape, map.shape, map.offset);
//         PBiCGStab3(x, pp, ss, alpha, omega, pdm, map, block_dim);
//         PBiCGStab4(r, s, t, omega, pdm, map, block_dim);

//         rrho = rho;

//         err = sqrt(MVDevCall::EuclideanNormSq(r, pdm, map, block_dim)) / map.size;
//         it ++;
//     } while (it < maxit && err > tol);
// }

}
