#include <math.h>
#include "structEqL2.h"
#include "MVL2.h"

namespace Falm {

void devL2_Struct3d7p_MV(Matrix<double> &a, Matrix<double> &x, Matrix<double> &ax, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    cpm.CPML2dev_IExchange6Face(a.dev.ptr, pdom, 1, 0, buffer, HDCType::Device, req);
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);
    devL0_Struct3d7p_MV(a, x, ax, pdom, inner_map, block_dim);

    cpm.CPML2_Wait6Face(req);
    cpm.CPML2dev_PostExchange6Face(a.dev.ptr, pdom, buffer, req);

    if (cpm.neighbour[0] >= 0) {
        Mapper emap(boundary_shape[0], boundary_offset[0]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, emap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[1] >= 0) {
        Mapper wmap(boundary_shape[1], boundary_offset[1]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, wmap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[2] >= 0) {
        Mapper nmap(boundary_shape[2], boundary_offset[2]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, nmap, dim3(8, 1, 8));
    }
    if (cpm.neighbour[3] >= 0) {
        Mapper smap(boundary_shape[3], boundary_offset[3]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, smap, dim3(8, 1 ,8));
    }
    if (cpm.neighbour[4] >= 0) {
        Mapper tmap(boundary_shape[4], boundary_offset[4]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, tmap, dim3(8, 8, 1));
    }
    if (cpm.neighbour[5] >= 0) {
        Mapper bmap(boundary_shape[5], boundary_offset[5]);
        devL0_Struct3d7p_MV(a, x, ax, pdom, bmap, dim3(8, 8, 1));
    }
}

void devL2_Struct3d7p_Res(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    cpm.CPML2dev_IExchange6Face(a.dev.ptr, pdom, 1, 0, buffer, HDCType::Device, req);
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);
    devL0_Struct3d7p_Res(a, x, b, r, pdom, inner_map, block_dim);

    cpm.CPML2_Wait6Face(req);
    cpm.CPML2dev_PostExchange6Face(a.dev.ptr, pdom, buffer, req);

    if (cpm.neighbour[0] >= 0) {
        Mapper emap(boundary_shape[0], boundary_offset[0]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, emap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[1] >= 0) {
        Mapper wmap(boundary_shape[1], boundary_offset[1]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, wmap, dim3(1, 8, 8));
    }
    if (cpm.neighbour[2] >= 0) {
        Mapper nmap(boundary_shape[2], boundary_offset[2]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, nmap, dim3(8, 1, 8));
    }
    if (cpm.neighbour[3] >= 0) {
        Mapper smap(boundary_shape[3], boundary_offset[3]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, smap, dim3(8, 1 ,8));
    }
    if (cpm.neighbour[4] >= 0) {
        Mapper tmap(boundary_shape[4], boundary_offset[4]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, tmap, dim3(8, 8, 1));
    }
    if (cpm.neighbour[5] >= 0) {
        Mapper bmap(boundary_shape[5], boundary_offset[5]);
        devL0_Struct3d7p_Res(a, x, b, r, pdom, bmap, dim3(8, 8, 1));
    }
}

void L2EqSolver::devL2_Struct3d7p_Jacobi(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    Mapper gmap(global, Gd);
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<double> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpm.CPML2dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0, buffer, HDCType::Device, req);

        devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6Face(a.dev.ptr, pdom, buffer, req);

        devL2_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);

        devL2_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
        err = sqrt(devL2_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::devL2_Struct3d7p_JacobiPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<double> xp(x.shape.x, x.shape.y, HDCType::Device, x.label);
    int __it = 0;
    do {
        xp.cpy(x, HDCType::Device);
        cpm.CPML2dev_IExchange6Face(xp.dev.ptr, pdom, 1, 0, buffer, HDCType::Device, req);

        devL0_Struct3d7p_JacobiSweep(a, x, xp, b, pdom, inner_map, block_dim);

        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6Face(a.dev.ptr, pdom, buffer, req);

        devL2_Struct3d7p_JacobiSweepBoundary(a, x, xp, b, pdom, cpm, boundary_shape, boundary_offset);
        
        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::devL2_Struct3d7p_SOR(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    Mapper gmap(global, Gd);
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    it = 0;
    do {
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, 1, 0, buffer, HDCType::Device, req);
        devL0_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Black, pdom, inner_map, block_dim);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, buffer, req);
        devL2_Struct3d7p_SORSweepBoundary(a, x, b, relax_factor, Color::Black, pdom, cpm, boundary_shape, boundary_offset);

        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0, buffer, HDCType::Device, req);
        devL0_Struct3d7p_SORSweep(a, x, b, relax_factor, Color::Red, pdom, inner_map, block_dim);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, buffer, req);
        devL2_Struct3d7p_SORSweepBoundary(a, x, b, relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);

        devL2_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
        err = sqrt(devL2_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

void L2EqSolver::devL2_Struct3d7p_SORPC(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    CPMBuffer<double> *buffer;
    MPI_Request *req;
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    int __it = 0;
    do {
        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, 1, 0, buffer, HDCType::Device, req);
        devL0_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Black, pdom, inner_map, block_dim);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, pdom, Color::Red, buffer, req);
        devL2_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Black, pdom, cpm, boundary_shape, boundary_offset);

        cpm.CPML2dev_IExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, 1, 0, buffer, HDCType::Device, req);
        devL0_Struct3d7p_SORSweep(a, x, b, pc_relax_factor, Color::Red, pdom, inner_map, block_dim);
        cpm.CPML2_Wait6Face(req);
        cpm.CPML2dev_PostExchange6ColoredFace(x.dev.ptr, pdom, Color::Black, buffer, req);
        devL2_Struct3d7p_SORSweepBoundary(a, x, b, pc_relax_factor, Color::Red, pdom, cpm, boundary_shape, boundary_offset);

        __it ++;
    } while (__it < pc_maxit);
}

void L2EqSolver::devL2_Struct3d7p_PBiCGStab(Matrix<double> &a, Matrix<double> &x, Matrix<double> &b, Matrix<double> &r, Mapper &global, Mapper &pdom, dim3 block_dim, CPM &cpm) {
    Mapper gmap(global, Gd);
    Mapper map(pdom, Gd);
    uint3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.setRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, pdom);

    Mapper inner_map(inner_shape, inner_offset);

    Matrix<double> rr(pdom.shape, 1, HDCType::Device, 101);
    Matrix<double>  p(pdom.shape, 1, HDCType::Device, 102);
    Matrix<double>  q(pdom.shape, 1, HDCType::Device, 103);
    Matrix<double>  s(pdom.shape, 1, HDCType::Device, 104);
    Matrix<double> pp(pdom.shape, 1, HDCType::Device, 105);
    Matrix<double> ss(pdom.shape, 1, HDCType::Device, 106);
    Matrix<double>  t(pdom.shape, 1, HDCType::Device, 107);
    double rho, rrho, alpha, beta, omega;

    devL2_Struct3d7p_Res(a, x, b, r, pdom, block_dim, cpm);
    err = sqrt(devL2_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;

    rr.cpy(r, HDCType::Device);
    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        if (err < tol) {
            break;
        }

        rho = devL2_DotProduct(r, rr, pdom, block_dim, cpm);
        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.cpy(r, HDCType::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            devL0_PBiCGStab1(p, q, r, beta, omega, pdom, map, block_dim);
        }
        pp.clear(HDCType::Device);
        devL2_Struct3d7p_Precondition(a, pp, p, pdom, block_dim, cpm);
        devL2_Struct3d7p_MV(a, pp, q, pdom, block_dim, cpm);
        alpha = rho / devL2_DotProduct(rr, q, pdom, block_dim, cpm);

        devL0_PBiCGStab2(s, q, r, alpha, pdom, map, block_dim);
        ss.clear(HDCType::Device);
        devL2_Struct3d7p_Precondition(a, ss, s, pdom, block_dim, cpm);
        devL2_Struct3d7p_MV(a, ss, t, pdom, block_dim, cpm);
        omega = devL2_DotProduct(t, s, pdom, block_dim, cpm) / devL2_DotProduct(t, t, pdom, block_dim, cpm);

        devL0_PBiCGStab3(x, pp, ss, alpha, omega, pdom, map, block_dim);
        devL0_PBiCGStab4(r, s, t, omega, pdom, map, block_dim);

        rrho = rho;

        err = sqrt(devL2_Norm2Sq(r, pdom, block_dim, cpm)) / gmap.size;
        it ++;
    } while (it < maxit && err > tol);
}

}
