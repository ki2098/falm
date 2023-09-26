#include <math.h>
#include "structEqL2.h"
#include "MVL2.h"

namespace Falm {

void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    CPMOp<REAL> cpmop(cpm);
    cpmop.CPML2Dev_IExchange6Face(x.dev.ptr, pdom, 1, 0);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);
    L0Dev_Struct3d7p_MV(a, x, ax, pdom, inner_map, block_dim);

    cpmop.CPML2_Wait6Face();
    cpmop.CPML2Dev_PostExchange6Face();

    if (cpm.neighbour[0] >= 0) {
        Mapper emap(boundary_shape[0], boundary_offset[0]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, emap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[1] >= 0) {
        Mapper wmap(boundary_shape[1], boundary_offset[1]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, wmap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[2] >= 0) {
        Mapper nmap(boundary_shape[2], boundary_offset[2]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, nmap, dim3(8, 1, 8));
    }
    if (cpm.neighbour[3] >= 0) {
        Mapper smap(boundary_shape[3], boundary_offset[3]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, smap, dim3(8, 1 ,8));
    }
    if (cpm.neighbour[4] >= 0) {
        Mapper tmap(boundary_shape[4], boundary_offset[4]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, tmap, dim3(8, 8, 1));
    }
    if (cpm.neighbour[5] >= 0) {
        Mapper bmap(boundary_shape[5], boundary_offset[5]);
        L0Dev_Struct3d7p_MV(a, x, ax, pdom, bmap, dim3(8, 8, 1));
    }
}

void L2Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    CPMOp<REAL> cpmop(cpm);
    cpmop.CPML2Dev_IExchange6Face(x.dev.ptr, pdom, 1, 0);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);
    L0Dev_Struct3d7p_Res(a, x, b, r, pdom, inner_map, block_dim);

    cpmop.CPML2_Wait6Face();
    cpmop.CPML2Dev_PostExchange6Face();

    if (cpm.neighbour[0] >= 0) {
        Mapper emap(boundary_shape[0], boundary_offset[0]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, emap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[1] >= 0) {
        Mapper wmap(boundary_shape[1], boundary_offset[1]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, wmap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[2] >= 0) {
        Mapper nmap(boundary_shape[2], boundary_offset[2]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, nmap, dim3(8, 1, 8));
    }
    if (cpm.neighbour[3] >= 0) {
        Mapper smap(boundary_shape[3], boundary_offset[3]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, smap, dim3(8, 1 ,8));
    }
    if (cpm.neighbour[4] >= 0) {
        Mapper tmap(boundary_shape[4], boundary_offset[4]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, tmap, dim3(8, 8, 1));
    }
    if (cpm.neighbour[5] >= 0) {
        Mapper bmap(boundary_shape[5], boundary_offset[5]);
        L0Dev_Struct3d7p_Res(a, x, b, r, pdom, bmap, dim3(8, 8, 1));
    }
}

void L2EqSolver::L2Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper gmap(global, Gd);
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        L2Dev_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);

        L2Dev_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
        err = sqrt(L2Dev_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::L2Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    INT __it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        L2Dev_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);
        
        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::L2Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper gmap(global, Gd);
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    it = 0;
    do {
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdom, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, relax_factor, Color::Black, pdom, cpm, boundary_shape, boundary_offset);

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdom, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);

        L2Dev_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
        err = sqrt(L2Dev_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::L2Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    INT __it = 0;
    do {
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdom, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Black, pdom, cpm, boundary_shape, boundary_offset);

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red, pdom, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);

        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::L2Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper gmap(global, Gd);
    Mapper map(pdom, Gd);

    Matrix<REAL> rr(pdom.shape, 1, HDCType::Device, 101);
    Matrix<REAL>  p(pdom.shape, 1, HDCType::Device, 102);
    Matrix<REAL>  q(pdom.shape, 1, HDCType::Device, 103);
    Matrix<REAL>  s(pdom.shape, 1, HDCType::Device, 104);
    Matrix<REAL> pp(pdom.shape, 1, HDCType::Device, 105);
    Matrix<REAL> ss(pdom.shape, 1, HDCType::Device, 106);
    Matrix<REAL>  t(pdom.shape, 1, HDCType::Device, 107);
    REAL rho, rrho, alpha, beta, omega;

    L2Dev_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
    err = sqrt(L2Dev_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;

    rr.cpy(r, HDCType::Device);
    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        if (err < tol) {
            break;
        }

        rho = L2Dev_DotProduct(r, rr, pdom, block_dim, cpm);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            L0Dev_PBiCGStab1(p, q, r, beta, omega, pdom, map, block_dim);
        }
        pp.clear(HDCType::Device);
        L2Dev_Struct3d7p_Precondition(a, pp, p, pdom, block_dim, cpm);
        L2Dev_Struct3d7p_MV(a, pp, q, pdom, block_dim, cpm);
        alpha = rho / L2Dev_DotProduct(rr, q, pdom, block_dim, cpm);

        L0Dev_PBiCGStab2(s, q, r, alpha, pdom, map, block_dim);
        ss.clear(HDCType::Device);
        L2Dev_Struct3d7p_Precondition(a, ss, s, pdom, block_dim, cpm);
        L2Dev_Struct3d7p_MV(a, ss, t, pdom, block_dim, cpm);
        omega = L2Dev_DotProduct(t, s, pdom, block_dim, cpm) / L2Dev_DotProduct(t, t, pdom, block_dim, cpm);

        L0Dev_PBiCGStab3(x, pp, ss, alpha, omega, pdom, map, block_dim);
        L0Dev_PBiCGStab4(r, s, t, omega, pdom, map, block_dim);

        rrho = rho;

        err = sqrt(L2Dev_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

}
