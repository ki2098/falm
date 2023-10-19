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
    Mapper       &pdm,
    dim3          block_dim,
    CPMBase      &cpm,
    STREAM       *stream
) {
    CPMOp<REAL> ucpm(&cpm);
    CPMOp<REAL> vcpm(&cpm);
    CPMOp<REAL> wcpm(&cpm);
    CPMOp<REAL> nutcpm(&cpm);

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0,0), pdm, 2, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0,1), pdm, 2, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0,2), pdm, 2, 0, 2, stream);
    nutcpm.CPML2Dev_IExchange6Face(&nut.dev(0), pdm, 1, 0, 3, stream);

    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 2, Mapper(pdm, cpm.gc));


    L0Dev_Cartesian3d_FSCalcPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Mapper(inner_shape, inner_offset), block_dim);

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
            L0Dev_Cartesian3d_FSCalcPseudoU(un, u, uu, ua, nut, kx, g, ja, ff, pdm, Mapper(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
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
    Mapper       &pdm,
    dim3          block_dim,
    CPMBase      &cpm,
    STREAM       *stream
) {
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Mapper(pdm, cpm.gc - 1));
    Matrix<REAL> uc(pdm.shape, 3, HDCType::Device, "contra u at grid");

    CPMOp<REAL> ucpm(&cpm);
    CPMOp<REAL> vcpm(&cpm);
    CPMOp<REAL> wcpm(&cpm);

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0,0), pdm, 1, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0,1), pdm, 1, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0,2), pdm, 1, 0, 2, stream);

    L0Dev_Cartesian3d_UtoCU(u, uc, kx, ja, pdm, Mapper(inner_shape, inner_offset), block_dim);

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
            L0Dev_Cartesian3d_UtoCU(u, uc, kx, ja, pdm, Mapper(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }

    Mapper uumap(pdm, cpm.gc);
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
    Mapper       &pdm,
    dim3          block_dim,
    CPMBase      &cpm,
    STREAM       *stream
) {
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Mapper(pdm, cpm.gc));

    CPMOp<REAL> pcpm(&cpm);

    pcpm.CPML2Dev_IExchange6Face(p.dev.ptr, pdm, 1, 0, 0, stream);

    L0Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, Mapper(inner_shape, inner_offset), block_dim);

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
            L0Dev_Cartesian3d_ProjectPGrid(u, ua, p, kx, pdm, Mapper(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
        }
    }
    if (stream) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (cpm.validNeighbour(fid)) {
                falmWaitStream(stream[fid]);
            }
        }
    }

    Mapper uumap(pdm, cpm.gc);
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
    Mapper       &pdm,
    dim3          block_dim,
    CPMBase      &cpm,
    STREAM       *stream
) {
    if (SGSModel == SGSType::Empty) {
        return;
    }
    INTx3 inner_shape, inner_offset, boundary_shape[6], boundary_offset[6];
    cpm.set6Region(inner_shape, inner_offset, boundary_shape, boundary_offset, 1, Mapper(pdm, cpm.gc));

    CPMOp<REAL> ucpm;
    CPMOp<REAL> vcpm;
    CPMOp<REAL> wcpm;

    ucpm.CPML2Dev_IExchange6Face(&u.dev(0, 0), pdm, 1, 0, 0, stream);
    vcpm.CPML2Dev_IExchange6Face(&u.dev(0, 1), pdm, 1, 0, 1, stream);
    wcpm.CPML2Dev_IExchange6Face(&u.dev(0, 2), pdm, 1, 0, 2, stream);

    L0Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, Mapper(inner_shape, inner_offset), block_dim);

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
            L0Dev_Cartesian3d_SGS(u, nut, x, kx, ja, pdm, Mapper(boundary_shape[fid], boundary_offset[fid]), __block, fstream);
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