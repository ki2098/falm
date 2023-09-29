#include <math.h>
#include "structEqL2.h"
#include "MVL2.h"

namespace Falm {

void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    CPMOp<REAL> cpmop(cpm);
    cpmop.CPML2Dev_IExchange6Face(x.dev.ptr, pdm, 1, 0);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);
    L0Dev_Struct3d7p_MV(a, x, ax, pdm, inner_map, block_dim);

    cpmop.CPML2_Wait6Face();
    cpmop.CPML2Dev_PostExchange6Face();

    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            Mapper map(boundary_shape[fid], boundary_offset[fid]);
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Struct3d7p_MV(a, x, ax, pdm, map, __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    falmWaitStream();
}

void L2Dev_Struct3d7p_Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    CPMOp<REAL> cpmop(cpm);
    cpmop.CPML2Dev_IExchange6Face(x.dev.ptr, pdm, 1, 0);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);
    L0Dev_Struct3d7p_Res(a, x, b, r, pdm, inner_map, block_dim);

    cpmop.CPML2_Wait6Face();
    cpmop.CPML2Dev_PostExchange6Face();

    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            Mapper map(boundary_shape[fid], boundary_offset[fid]);
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Struct3d7p_Res(a, x, b, r, pdm, map, __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    falmWaitStream();
}

void L2EqSolver::L2Dev_Struct3d7p_Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    Mapper gmap(global, Gd);
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, "Jacobi" + x.name + "Previous");
    it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdm, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdm, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        L2Dev_Struct3d7p_Res(a, x, b, r, pdm, block_dim, cpm);
        err = sqrt(L2Dev_Norm2Sq(r, pdm, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::L2Dev_Struct3d7p_JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<REAL> xp(x.shape.x, x.shape.y, HDCType::Device, "Jacobi" + x.name + "Previous");
    INT __it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpmop.CPML2Dev_IExchange6Face(xp.dev.ptr, pdm, 1, 0);

        L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdm, inner_map, block_dim);

        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6Face();

        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_JacobiSweep(a, x, xp, b, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();
        
        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::L2Dev_Struct3d7p_SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    Mapper gmap(global, Gd);
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);

    it = 0;
    do {
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdm, Color::Red, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdm, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdm, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdm, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        L2Dev_Struct3d7p_Res(a, x, b, r, pdm, block_dim, cpm);
        err = sqrt(L2Dev_Norm2Sq(r, pdm, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::L2Dev_Struct3d7p_SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    CPMOp<REAL> cpmop(cpm);
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdm);

    Mapper inner_map(inner_shape, inner_offset);

    INT __it = 0;
    do {
        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdm, Color::Red, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdm, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        cpmop.CPML2Dev_IExchange6ColoredFace(x.dev.ptr, pdm, Color::Black, 1, 0);
        L0Dev_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red, pdm, inner_map, block_dim);
        cpmop.CPML2_Wait6Face();
        cpmop.CPML2Dev_PostExchange6ColoredFace();
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                Mapper map(boundary_shape[fid], boundary_offset[fid]);
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                L0Dev_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdm, map, __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::L2Dev_Struct3d7p_PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, Mapper &global, Mapper &pdm, dim3 block_dim, CPMBase &cpm, STREAM *stream) {
    Mapper gmap(global, Gd);
    Mapper map(pdm, Gd);

    Matrix<REAL> rr(pdm.shape, 1, HDCType::Device, "PBiCGStab rr");
    Matrix<REAL>  p(pdm.shape, 1, HDCType::Device, "PBiCGStab  p");
    Matrix<REAL>  q(pdm.shape, 1, HDCType::Device, "PBiCGStab  q");
    Matrix<REAL>  s(pdm.shape, 1, HDCType::Device, "PBiCGStab  s");
    Matrix<REAL> pp(pdm.shape, 1, HDCType::Device, "PBiCGStab pp");
    Matrix<REAL> ss(pdm.shape, 1, HDCType::Device, "PBiCGStab ss");
    Matrix<REAL>  t(pdm.shape, 1, HDCType::Device, "PBiCGStab  t");
    REAL rho, rrho, alpha, beta, omega;

    L2Dev_Struct3d7p_Res(a, x, b, r, pdm, block_dim, cpm);
    err = sqrt(L2Dev_Norm2Sq(r, pdm, block_dim, cpm)) / gmap.size;

    rr.cpy(r, HDCType::Device);
    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        if (err < tol) {
            break;
        }

        rho = L2Dev_DotProduct(r, rr, pdm, block_dim, cpm);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            L0Dev_PBiCGStab1(p, q, r, beta, omega, pdm, map, block_dim);
        }
        pp.clear(HDCType::Device);
        L2Dev_Struct3d7p_Precondition(a, pp, p, pdm, block_dim, cpm);
        L2Dev_Struct3d7p_MV(a, pp, q, pdm, block_dim, cpm);
        alpha = rho / L2Dev_DotProduct(rr, q, pdm, block_dim, cpm);

        L0Dev_PBiCGStab2(s, q, r, alpha, pdm, map, block_dim);
        ss.clear(HDCType::Device);
        L2Dev_Struct3d7p_Precondition(a, ss, s, pdm, block_dim, cpm);
        L2Dev_Struct3d7p_MV(a, ss, t, pdm, block_dim, cpm);
        omega = L2Dev_DotProduct(t, s, pdm, block_dim, cpm) / L2Dev_DotProduct(t, t, pdm, block_dim, cpm);

        L0Dev_PBiCGStab3(x, pp, ss, alpha, omega, pdm, map, block_dim);
        L0Dev_PBiCGStab4(r, s, t, omega, pdm, map, block_dim);

        rrho = rho;

        err = sqrt(L2Dev_Norm2Sq(r, pdm, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

}
