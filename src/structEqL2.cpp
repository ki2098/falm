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
    L0Dev_Struct3d7p_MV(a, x, ax, pdom, inner_map, block_dim, cudaStreamPerThread);

    cpmop.CPML2_Wait6Face();
    cpmop.CPML2Dev_PostExchange6Face();

    
    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            Mapper map(boundary_shape[fid], boundary_offset[fid]);
            // falmCreateStream(&boundaryStream[fid]);
            L0Dev_Struct3d7p_MV(a, x, ax, pdom, map, block_dim, boundaryStream[fid]);
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            falmStreamSync(boundaryStream[fid]);
            // falmDestroyStream(boundaryStream[fid]);
        }
    }
    falmStreamSync(cudaStreamPerThread);
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

    
    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            Mapper map(boundary_shape[fid], boundary_offset[fid]);
            // falmCreateStream(&boundaryStream[fid]);
            L0Dev_Struct3d7p_Res(a, x, b, r, pdom, map, block_dim, boundaryStream[fid]);
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            falmStreamSync(boundaryStream[fid]);
            // falmDestroyStream(boundaryStream[fid]);
        }
    }
    falmStreamSync(cudaStreamPerThread);
}

void L2EqSolver::L2Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper gmap(global, Gd);
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, x.name + " previous");
    it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        L2Dev_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);
        falmStreamSync(cudaStreamPerThread);

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

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, x.name + " previous");
    INT __it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        L2Dev_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);
        falmStreamSync(cudaStreamPerThread);
        
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
        falmStreamSync(cudaStreamPerThread);

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdom, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);
        falmStreamSync(cudaStreamPerThread);

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
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdom, inner_map, block_dim, cudaStreamPerThread);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Black, pdom, cpm, boundary_shape, boundary_offset);
        // falmStreamSync(cudaStreamPerThread);

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red, pdom, inner_map, block_dim, cudaStreamPerThread);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        L2Dev_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);
        // falmStreamSync(cudaStreamPerThread);

        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::L2Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPMBase &cpm) {
    Mapper gmap(global, Gd);
    Mapper map(pdom, Gd);

    Matrix<REAL> rr(pdom.shape, 1, HDCType::Device, "PBiCGStab rr");
    Matrix<REAL>  p(pdom.shape, 1, HDCType::Device, "PBiCGStab p" );
    Matrix<REAL>  q(pdom.shape, 1, HDCType::Device, "PBiCGStab q" );
    Matrix<REAL>  s(pdom.shape, 1, HDCType::Device, "PBiCGStab s" );
    Matrix<REAL> pp(pdom.shape, 1, HDCType::Device, "PBiCGStab pp");
    Matrix<REAL> ss(pdom.shape, 1, HDCType::Device, "PBiCGStab ss");
    Matrix<REAL>  t(pdom.shape, 1, HDCType::Device, "PBiCGStab t" );
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
