#include <math.h>
#include "FalmEq.h"
#include "MV.h"
#include "profiler.h"

// extern Cprof::cprof_Profiler pprofiler;

namespace Falm {

// void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, CPMBase &cpm, dim3 block_dim, STREAM *stream) {
//     Region &pdm = cpm.pdm_list[cpm.rank];
//     CPMComm<REAL> cpmop(&cpm);
//     cpmop.IExchange6Face(x.dev.ptr, 1, 0, 0);
//     INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
//     cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

//     // Mapper inner_map(inner_shape, inner_offset);
//     MVDevCall::MVMult(a, x, ax, pdm, Region(inner_shape, inner_offset), block_dim);

//     cpmop.CPML2_Wait6Face();
//     cpmop.PostExchange6Face();

//     for (INT fid = 0; fid < 6; fid ++) {
//         if (cpm.neighbour[fid] >= 0) {
//             dim3 __block(
//                 (fid / 2 == 0)? 1U : 8U,
//                 (fid / 2 == 1)? 1U : 8U,
//                 (fid / 2 == 2)? 1U : 8U
//             );
//             // Mapper map(boundary_shape[fid], boundary_offset[fid]);
//             STREAM fstream = (stream)? stream[fid] : (STREAM)0;
//             MVDevCall::MVMult(a, x, ax, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
//         }
//     }
//     if (stream) {
//         for (INT fid = 0; fid < 6; fid ++) {
//             if (cpm.neighbour[fid] >= 0) {
//                 falmWaitStream(stream[fid]);
//             }
//         }
//     }
//     falmWaitStream();
// }

void FalmEq::Res(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, Matrix<Real> &r, CPM &cpm, dim3 block_dim, Stream *stream) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    CPMComm<Real> cpmop(&cpm);
    cpmop.IExchange6Face(x.dev.ptr, 1, 0, 0);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    FalmEqDevCall::Res(a, x, b, r, pdm, Region(inner_shape, inner_offset), block_dim);

    cpmop.Wait6Face();
    cpmop.PostExchange6Face();

    for (Int fid = 0; fid < 6; fid ++) {
        if (cpm.neighbour[fid] >= 0) {
            dim3 __block(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            // Mapper map(boundary_shape[fid], boundary_offset[fid]);
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            FalmEqDevCall::Res(a, x, b, r, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    
    falmWaitStream();
}

void FalmEq::Jacobi(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, Matrix<Real> &r, CPM &cpm, dim3 block_dim, Stream *stream) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    Region gmap(global.shape, cpm.gc);
    
    CPMComm<Real> cpmop(&cpm);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    Region inner_map(inner_shape, inner_offset);

    // Matrix<REAL> xp(x.shape[0], x.shape[1], HDC::Device, "Jacobi" + x.name + "Previous");
    it = 0;
    do {
        xp.copy(x, HDC::Device);
        cpmop.IExchange6Face(xp.dev.ptr, 1, 0, 0);

        JacobiSweep(a, x, xp, b, pdm, inner_map, block_dim);

        cpmop.Wait6Face();
        cpmop.PostExchange6Face();

        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                JacobiSweep(a, x, xp, b, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
            
        Res(a, x, b, r, cpm, block_dim);
        err = sqrt(FalmMV::EuclideanNormSq(r, cpm, block_dim)) / gmap.size;
        it ++;
        falmWaitStream();
    } while (it < maxit && err > tol);
}

void FalmEq::JacobiPC(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, CPM &cpm, dim3 block_dim, Stream *stream) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Matrix<Real> xp(x.shape[0], x.shape[1], HDC::Device, "Jacobi" + x.name + "Previous");

    CPMComm<Real> cpmop(&cpm);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));
    
    Region inner_map(inner_shape, inner_offset);
        
    Int __it = 0;
    do {
        xp.copy(x, HDC::Device);
        cpmop.IExchange6Face(xp.dev.ptr, 1, 0, 0);
    
        JacobiSweep(a, x, xp, b, pdm, inner_map, block_dim);
    
        cpmop.Wait6Face();
        cpmop.PostExchange6Face();
    
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                JacobiSweep(a, x, xp, b, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        __it ++;
        falmWaitStream();
    } while (__it < pc_maxit);
}

void FalmEq::SOR(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, Matrix<Real> &r, CPM &cpm, dim3 block_dim, Stream *stream) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region &global = cpm.global;
    Region  gmap(global.shape, cpm.gc);
    CPMComm<Real> cpmop(&cpm);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    Region inner_map(inner_shape, inner_offset);

    it = 0;
    do {
        cpmop.IExchange6ColoredFace(x.dev.ptr, Color::Red, 1, 0, 0);
        SORSweep(a, x, b, relax_factor, Color::Black, pdm, inner_map, block_dim);
        cpmop.Wait6Face();
        cpmop.PostExchange6ColoredFace();
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                SORSweep(a, x, b, relax_factor, Color::Black, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }

        falmWaitStream();

        cpmop.IExchange6ColoredFace(x.dev.ptr, Color::Black, 1, 0, 0);
        SORSweep(a, x, b, relax_factor, Color::Red, pdm, inner_map, block_dim);
        cpmop.Wait6Face();
        cpmop.PostExchange6ColoredFace();
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                SORSweep(a, x, b, relax_factor, Color::Red, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }

        Res(a, x, b, r, cpm, block_dim);
        err = sqrt(FalmMV::EuclideanNormSq(r, cpm, block_dim)) / gmap.size;
        it ++;
        falmWaitStream();
    } while (it < maxit && err > tol);
}

void FalmEq::SORPC(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, CPM &cpm, dim3 block_dim, Stream *stream) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    CPMComm<Real> cpmop(&cpm);
    Int3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    Region inner_map(inner_shape, inner_offset);

    Int __it = 0;
    do {
        cpmop.IExchange6ColoredFace(x.dev.ptr, Color::Red, 1, 0, 0);
        SORSweep(a, x, b, pc_relax_factor, Color::Black, pdm, inner_map, block_dim);
        cpmop.Wait6Face();
        cpmop.PostExchange6ColoredFace();
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                SORSweep(a, x, b, pc_relax_factor, Color::Black, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();

        cpmop.IExchange6ColoredFace(x.dev.ptr, Color::Black, 1, 0, 0);
        SORSweep(a, x, b, pc_relax_factor, Color::Red, pdm, inner_map, block_dim);
        cpmop.Wait6Face();
        cpmop.PostExchange6ColoredFace();
        for (Int fid = 0; fid < 6; fid ++) {
            if (cpm.neighbour[fid] >= 0) {
                dim3 __block(
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
                );
                // Mapper map(boundary_shape[fid], boundary_offset[fid]);
                Stream fstream = (stream)? stream[fid] : (Stream)0;
                SORSweep(a, x, b, pc_relax_factor, Color::Red, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (Int fid = 0; fid < 6; fid ++) {
                if (cpm.neighbour[fid] >= 0) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
        falmWaitStream();
        __it ++;
    } while (__it < pc_maxit);
}

void FalmEq::PBiCGStab(Matrix<Real> &a, Matrix<Real> &x, Matrix<Real> &b, Matrix<Real> &r, CPM &cpm, dim3 block_dim, Stream *stream) {
    // pprofiler.startEvent("PBiCGStab");
    // printf("PBiCGStab run\n");
    // pprofiler.startEvent("PBiCGStab init");
    Region &global = cpm.global;
    Region &pdm = cpm.pdm_list[cpm.rank];
    Region gmap(global.shape, cpm.gc);
    Region map(pdm.shape, cpm.gc);

    // Matrix<REAL> rr(pdm.shape, 1, HDC::Device, "PBiCGStab rr");
    // Matrix<REAL>  p(pdm.shape, 1, HDC::Device, "PBiCGStab  p");
    // Matrix<REAL>  q(pdm.shape, 1, HDC::Device, "PBiCGStab  q");
    // Matrix<REAL>  s(pdm.shape, 1, HDC::Device, "PBiCGStab  s");
    // Matrix<REAL> pp(pdm.shape, 1, HDC::Device, "PBiCGStab pp");
    // Matrix<REAL> ss(pdm.shape, 1, HDC::Device, "PBiCGStab ss");
    // Matrix<REAL>  t(pdm.shape, 1, HDC::Device, "PBiCGStab  t");
    Real rho, rrho, alpha, beta, omega;
    // pprofiler.endEvent("PBiCGStab init");

    // size_t freebyte, totalbyte;
    // cudaMemGetInfo(&freebyte, &totalbyte);
    // printf("\nrank %d: free %lf, total %lf\n", cpm.rank, freebyte / (1024. * 1024.), totalbyte / (1024. * 1024.));

    // pprofiler.startEvent("||b - Ax||");
    Res(a, x, b, r, cpm, block_dim);
    err = sqrt(FalmMV::EuclideanNormSq(r, cpm, block_dim)) / gmap.size;
    // pprofiler.endEvent("||b - Ax||");

    rr.copy(r, HDC::Device);
    rrho  = 1.0;
    alpha = 0.0;
    omega = 1.0;

    it = 0;
    do {
        // if (err < tol) {
        //     break;
        // }

        // pprofiler.startEvent("rho = r dor r*");
        rho = FalmMV::DotProduct(r, rr, cpm, block_dim);
        // pprofiler.endEvent("rho = r dor r*");

        if (fabs(rho) < __FLT_MIN__) {
            err = rho;
            break;
        }

        if (it == 0) {
            p.copy(r, HDC::Device);
        } else {
            beta = (rho * alpha) / (rrho * omega);
            // pprofiler.startEvent("p = r + beta * (p - omega * q) [1]");
            PBiCGStab1(p, q, r, beta, omega, pdm, map, block_dim);
            // pprofiler.endEvent("p = r + beta * (p - omega * q) [1]");
        }
        pp.clear(HDC::Device);
        // pprofiler.startEvent("Ap* ~ p");
        Precondition(a, pp, p, cpm, block_dim);
        // pprofiler.endEvent("Ap* ~ p");

        // pprofiler.startEvent("q = Ap*");
        FalmMV::MV(a, pp, q, cpm, block_dim);
        // pprofiler.endEvent("q = Ap*");

        // pprofiler.startEvent("alpha = rho / (r* dot q)");
        alpha = rho / FalmMV::DotProduct(rr, q, cpm, block_dim);
        // pprofiler.endEvent("alpha = rho / (r* dot q)");

        // pprofiler.startEvent("s = r - alpha * q [2]");
        PBiCGStab2(s, q, r, alpha, pdm, map, block_dim);
        // pprofiler.endEvent("s = r - alpha * q [2]");

        ss.clear(HDC::Device);

        // pprofiler.startEvent("As* ~ s");
        Precondition(a, ss, s, cpm, block_dim);
        // pprofiler.endEvent("As* ~ s");

        // pprofiler.startEvent("t = As*");
        FalmMV::MV(a, ss, t, cpm, block_dim);
        // pprofiler.endEvent("t = As*");

        // pprofiler.startEvent("omega = (t dot s) / (t dot t)");
        omega = FalmMV::DotProduct(t, s, cpm, block_dim) / FalmMV::DotProduct(t, t, cpm, block_dim);
        // pprofiler.endEvent("omega = (t dot s) / (t dot t)");

        // pprofiler.startEvent("x += alpha * p* + omega * s* [3]");
        PBiCGStab3(x, pp, ss, alpha, omega, pdm, map, block_dim);
        // pprofiler.endEvent("x += alpha * p* + omega * s* [3]");

        // pprofiler.startEvent("r = s - omega * t [4]");
        PBiCGStab4(r, s, t, omega, pdm, map, block_dim);
        // pprofiler.endEvent("r = s - omega * t [4]");

        rrho = rho;

        // pprofiler.startEvent("||b - Ax||");
        err = sqrt(FalmMV::EuclideanNormSq(r, cpm, block_dim)) / gmap.size;
        // pprofiler.endEvent("||b - Ax||");

        it ++;
    } while (it < maxit && err > tol);

    // pprofiler.endEvent("PBiCGStab");
}

}
