#include "FalmCFD.h"

namespace Falm {

void FalmCFD::FSPseudoU(
    Matrix<REAL> &un,
    Matrix<REAL> &u,
    Matrix<REAL> &uu,
    Matrix<REAL> &ua,
    Matrix<REAL> &nut,
    Matrix<REAL> &kx,
    Matrix<REAL> &g,
    Matrix<REAL> &ja,
    Matrix<REAL> &ff,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    if (cpm.size == 1) {
        Region  map(pdm.shape, cpm.gc);
        FalmCFDDevCall::FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, map, block_dim);
    } else {
        CPMComm<REAL> ucpm(&cpm);
        CPMComm<REAL> vcpm(&cpm);
        CPMComm<REAL> wcpm(&cpm);
        CPMComm<REAL> nutcpm(&cpm);

        ucpm.IExchange6Face(&u.dev(0,0), 2, 0, 0, stream);
        vcpm.IExchange6Face(&u.dev(0,1), 2, 0, 1, stream);
        wcpm.IExchange6Face(&u.dev(0,2), 2, 0, 2, stream);
        nutcpm.IExchange6Face(&nut.dev(0), 1, 0, 3, stream);

        INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
        cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 2, Region(pdm.shape, cpm.gc));


        FalmCFDDevCall::FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Region(inner_shape, inner_offset), block_dim);

        ucpm.Wait6Face();
        vcpm.Wait6Face();
        wcpm.Wait6Face();
        nutcpm.Wait6Face();

        ucpm.PostExchange6Face();
        vcpm.PostExchange6Face();
        wcpm.PostExchange6Face();
        nutcpm.PostExchange6Face();

        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                FalmCFDDevCall::FSPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.validNeighbour(fid)) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
    }
    falmWaitStream();
}

void FalmCFD::UtoUU(
    Matrix<REAL> &u,
    Matrix<REAL> &uu,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    Matrix<REAL> uc(pdm.shape, 3, HDCType::Device, "contra u at grid");
    if (cpm.size == 1) {
        Region map(pdm.shape, cpm.gc - 1);
        FalmCFDDevCall::UtoCU(u, uc, kx, ja, pdm, map, block_dim);
    } else {
        Region &pdm = cpm.pdm_list[cpm.rank];
        INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
        cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc - 1));

        CPMComm<REAL> ucpm(&cpm);
        CPMComm<REAL> vcpm(&cpm);
        CPMComm<REAL> wcpm(&cpm);

        ucpm.IExchange6Face(&u.dev(0,0), 1, 0, 0, stream);
        vcpm.IExchange6Face(&u.dev(0,1), 1, 0, 1, stream);
        wcpm.IExchange6Face(&u.dev(0,2), 1, 0, 2, stream);

        FalmCFDDevCall::UtoCU(u, uc, kx, ja, pdm, Region(inner_shape, inner_offset), block_dim);

        ucpm.Wait6Face();
        vcpm.Wait6Face();
        wcpm.Wait6Face();

        ucpm.PostExchange6Face();
        vcpm.PostExchange6Face();
        wcpm.PostExchange6Face();
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                FalmCFDDevCall::UtoCU(u, uc, kx, ja, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.validNeighbour(fid)) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
    }

    Region uumap(pdm.shape, cpm.gc);
    uumap.shape  += {1, 1, 1};
    uumap.offset -= {1, 1, 1};
    // uumap = uumap.transform(
    //     INT3{ 1,  1,  1},
    //     INT3{-1, -1, -1}
    // );
    FalmCFDDevCall::InterpolateCU(uu, uc, pdm, uumap, block_dim);

    falmWaitStream();
}

void FalmCFD::ProjectP(
    Matrix<REAL> &u,
    Matrix<REAL> &ua,
    Matrix<REAL> &uu,
    Matrix<REAL> &uua,
    Matrix<REAL> &p,
    Matrix<REAL> &kx,
    Matrix<REAL> &g,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    if (cpm.size == 1) {
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::ProjectPGrid(u, ua, p, kx, pdm, map, block_dim);
    } else {
        INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
        cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

        CPMComm<REAL> pcpm(&cpm);
        pcpm.IExchange6Face(p.dev.ptr, 1, 0, 0, stream);

        FalmCFDDevCall::ProjectPGrid(u, ua, p, kx, pdm, Region(inner_shape, inner_offset), block_dim);

        pcpm.Wait6Face();
        pcpm.PostExchange6Face();

        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                FalmCFDDevCall::ProjectPGrid(u, ua, p, kx, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.validNeighbour(fid)) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
    }

    Region uumap(pdm.shape, cpm.gc);
    uumap.shape  += {1, 1, 1};
    uumap.offset -= {1, 1, 1};
    // uumap = uumap.transform(
    //     INT3{ 1,  1,  1},
    //     INT3{-1, -1, -1}
    // );
    FalmCFDDevCall::ProjectPFace(uu, uua, p, g, pdm, uumap, block_dim);

    falmWaitStream();
}

void FalmCFD::SGS(
    Matrix<REAL> &u,
    Matrix<REAL> &nut,
    Matrix<REAL> &x,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    if (SGSModel == SGSType::Empty) {
        return;
    }
    Region &pdm = cpm.pdm_list[cpm.rank];
    if (cpm.size == 1) {
        Region map(pdm.shape, cpm.gc);
        FalmCFDDevCall::SGS(u, nut, x, kx, ja, pdm, map, block_dim);
    } else {
        INT3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
        cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

        CPMComm<REAL> ucpm;
        CPMComm<REAL> vcpm;
        CPMComm<REAL> wcpm;

        ucpm.IExchange6Face(&u.dev(0, 0), 1, 0, 0, stream);
        vcpm.IExchange6Face(&u.dev(0, 1), 1, 0, 1, stream);
        wcpm.IExchange6Face(&u.dev(0, 2), 1, 0, 2, stream);

        FalmCFDDevCall::SGS(u, nut, x, kx, ja, pdm, Region(inner_shape, inner_offset), block_dim);

        ucpm.Wait6Face();
        vcpm.Wait6Face();
        wcpm.Wait6Face();

        ucpm.PostExchange6Face();
        vcpm.PostExchange6Face();
        wcpm.PostExchange6Face();

        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                dim3 __block(
                    (fid / 2 == 0)? 1U : 8U,
                    (fid / 2 == 1)? 1U : 8U,
                    (fid / 2 == 2)? 1U : 8U
                );
                STREAM fstream = (stream)? stream[fid] : (STREAM)0;
                FalmCFDDevCall::SGS(u, nut, x, kx, ja, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
            }
        }
        if (stream) {
            for (INT fid = 0; fid < 6; fid ++) {
                if (cpm.validNeighbour(fid)) {
                    falmWaitStream(stream[fid]);
                }
            }
        }
    }
    falmWaitStream();
}

}