#ifndef FALM_STRUCTEQL2_H
#define FALM_STRUCTEQL2_H

#include <string>
#include "FalmEqDevCall.h"
#include "CPM.h"

namespace Falm {

// void L2Dev_Struct3d7p_MV(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &ax, CPMBase &cpm, dim3 block_dim, STREAM *stream = nullptr);



class FalmEq : public FalmEqDevCall {
public:
    Matrix<REAL> rr, p, q, s, pp, ss, t, xp;

    FalmEq() {}
    FalmEq(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) : 
        FalmEqDevCall(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor) 
    {}

    void init(FLAG _type, INT _maxit, REAL _tol, REAL _relax_factor, FLAG _pc_type = SolverType::Empty, INT _pc_maxit = 5, REAL _pc_relax_factor = 1.0) {
        FalmEqDevCall::init(_type, _maxit, _tol, _relax_factor, _pc_type, _pc_maxit, _pc_relax_factor);
    }

    void alloc(INT3 shape) {
        if (type == LSType::Jacobi || pc_type == LSType::Jacobi) {
            xp.alloc(shape, 1, HDC::Device, "Jacobi x<n> in Ax=b");
        }
        if (type == LSType::PBiCGStab) {
            rr.alloc(shape, 1, HDC::Device, "PBiCGStab rr");
             p.alloc(shape, 1, HDC::Device, "PBiCGStab  p");
             q.alloc(shape, 1, HDC::Device, "PBiCGStab  q");
             s.alloc(shape, 1, HDC::Device, "PBiCGStab  s");
            pp.alloc(shape, 1, HDC::Device, "PBiCGStab pp");
            ss.alloc(shape, 1, HDC::Device, "PBiCGStab ss");
             t.alloc(shape, 1, HDC::Device, "PBiCGStab  t");
        }
    }

    void release() {
        xp.release();
        rr.release();
         p.release();
         q.release();
         s.release();
        pp.release();
        ss.release();
         t.release();
    }

    void Jacobi(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void SOR(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void PBiCGStab(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void Solve(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr) {
        if (type == SolverType::Jacobi) {
            Jacobi(a, x, b, r, cpm, block_dim, stream);
        } else if (type == SolverType::SOR) {
            SOR(a, x, b, r, cpm, block_dim, stream);
        } else if (type == SolverType::PBiCGStab) {
            PBiCGStab(a, x, b, r, cpm, block_dim, stream);
        }
    }

public:
    static void Res(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, Matrix<REAL> &r, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void JacobiPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void SORPC(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr);
    void Precondition(Matrix<REAL> &a, Matrix<REAL> &x, Matrix<REAL> &b, CPM &cpm, dim3 block_dim, STREAM *stream = nullptr) {
        if (pc_type == SolverType::Jacobi) {
            JacobiPC(a, x, b, cpm, block_dim, stream);
        } else if (pc_type == SolverType::SOR) {
            SORPC(a, x, b, cpm, block_dim, stream);
        }
    }

public:
    static FLAG str2type(std::string str) {
        if (str == "PBiCGStab") {
            return LSType::PBiCGStab;
        } else if (str == "SOR") {
            return LSType::SOR;
        } else if (str == "Jacobi") {
            return LSType::Jacobi;
        } else {
            return LSType::Empty;
        }
    }

    static std::string type2str(FLAG lst) {
        if (lst == LSType::Empty) {
            return "Empty";
        } else if (lst == LSType::PBiCGStab) {
            return "PBiCGStab";
        } else if (lst == LSType::SOR) {
            return "SOR";
        } else if (lst == LSType::Jacobi) {
            return "Jacobi";
        } else {
            return "Not defined";
        }
    }
};

}

#endif
