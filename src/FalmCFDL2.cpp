#include "FalmCFDL2.h"

namespace Falm {

void L2CFD::L2Dev_Cartesian3d_FSCalcPseudoU(
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
    CPMComm<REAL> ucpm(&cpm);
    CPMComm<REAL> vcpm(&cpm);
    CPMComm<REAL> wcpm(&cpm);
    CPMComm<REAL> nutcpm(&cpm);

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0,0), 2, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0,1), 2, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0,2), 2, 0, 2, stream);
    nutcpm.CPML2Dev_IExchange6Face(&nut.dev(0), 1, 0, 3, stream);

    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 2, Region(pdm.shape, cpm.gc));


    L0Dev_Cartesian3d_FSCalcPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Region(inner_shape, inner_offset), block_dim);

    ucpm.CPML2_Wait6Face();
    vcpm.CPML2_Wait6Face();
    wcpm.CPML2_Wait6Face();
    nutcpm.CPML2_Wait6Face();

    ucpm.CPML2Dev_PostExchange6Face();
    vcpm.CPML2Dev_PostExchange6Face();
    wcpm.CPML2Dev_PostExchange6Face();
    nutcpm.CPML2Dev_PostExchange6Face();

    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.validNeighbour(fid)) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Cartesian3d_FSCalcPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    falmWaitStream();
}

void L2CFD::L2Dev_Cartesian3d_UtoUU(
    Matrix<REAL> &u,
    Matrix<REAL> &uu,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc - 1));
    Matrix<REAL> uc(pdm.shape, 3, HDCType::Device, "contra u at grid");

    CPMComm<REAL> ucpm(&cpm);
    CPMComm<REAL> vcpm(&cpm);
    CPMComm<REAL> wcpm(&cpm);

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0,0), 1, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0,1), 1, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0,2), 1, 0, 2, stream);

    L0Dev_Cartesian3d_UtoCU(u, uc, kx, ja, pdm, Region(inner_shape, inner_offset), block_dim);

    ucpm.CPML2_Wait6Face();
    vcpm.CPML2_Wait6Face();
    wcpm.CPML2_Wait6Face();

    ucpm.CPML2Dev_PostExchange6Face();
    vcpm.CPML2Dev_PostExchange6Face();
    wcpm.CPML2Dev_PostExchange6Face();
    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.validNeighbour(fid)) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Cartesian3d_UtoCU(u, uc, kx, ja, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }

    Region uumap(pdm.shape, cpm.gc);
    uumap = uumap.transform(
        INTx3{ 1,  1,  1},
        INTx3{-1, -1, -1}
    );
    L0Dev_Cartesian3d_InterpolateCU(uu, uc, pdm, uumap, block_dim);

    falmWaitStream();
}

void L2CFD::L2Dev_Cartesian3d_ProjectP(
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
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    CPMComm<REAL> pcpm(&cpm);

    pcpm.CPML2Dev_IExchange6Face(p.dev.ptr, 1, 0, 0, stream);

    L0Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, Region(inner_shape, inner_offset), block_dim);

    pcpm.CPML2_Wait6Face();
    pcpm.CPML2Dev_PostExchange6Face();

    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.validNeighbour(fid)) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }

    Region uumap(pdm.shape, cpm.gc);
    uumap = uumap.transform(
        INTx3{ 1,  1,  1},
        INTx3{-1, -1, -1}
    );
    L0Dev_Cartesian3d_ProjectPFace(uu, uua, p, g, pdm, uumap, block_dim);

    falmWaitStream();
}

void L2CFD::L2Dev_Cartesian3d_SGS(
    Matrix<REAL> &u,
    Matrix<REAL> &nut,
    Matrix<REAL> &x,
    Matrix<REAL> &kx,
    Matrix<REAL> &ja,
    CPMBase      &cpm,
    dim3          block_dim,
    STREAM       *stream
) {
    Region &pdm = cpm.pdm_list[cpm.rank];
    if (SGSModel == SGSType::Empty) {
        return;
    }
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Region(pdm.shape, cpm.gc));

    CPMComm<REAL> ucpm;
    CPMComm<REAL> vcpm;
    CPMComm<REAL> wcpm;

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0, 0), 1, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0, 1), 1, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0, 2), 1, 0, 2, stream);

    L0Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, Region(inner_shape, inner_offset), block_dim);

    ucpm.CPML2_Wait6Face();
    vcpm.CPML2_Wait6Face();
    wcpm.CPML2_Wait6Face();

    ucpm.CPML2Dev_PostExchange6Face();
    vcpm.CPML2Dev_PostExchange6Face();
    wcpm.CPML2Dev_PostExchange6Face();

    for (INT fid = 0; fid < 6; fid ++) {
        if (cpm.validNeighbour(fid)) {
            dim3 __block(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            L0Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, Region(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }
    falmWaitStream();
}

}