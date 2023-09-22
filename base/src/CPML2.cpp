#include "CPML2.h"

namespace Falm {

void CPM::CPML2_Wait6Face(MPI_Request *req) {
    CPML2_Waitall(nP2P, req, MPI_STATUSES_IGNORE);
}

void CPM::CPML2dev_IExchange6Face(double *data, Mapper &pdom, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, unsigned int buf_hdctype, MPI_Request *&req) {
    buffer = new CPMBuffer<double>[12];
    req    = new MPI_Request[12];
    nP2P   = 0;
    uint3 yz_inner_slice{thick, pdom.shape.y - Gdx2, pdom.shape.z - Gdx2};
    dim3 yz_block_dim(1, 8, 8);
    if (neighbour[0] >= 0) {
        buffer[nP2P].alloc(
            yz_inner_slice,
            uint3{pdom.shape.x - Gd - thick, Gd, Gd},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, yz_block_dim);
        buffer[nP2P + 1].alloc(
            yz_inner_slice,
            uint3{pdom.shape.x - Gd        , Gd, Gd},
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[0], neighbour[0] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[0], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[1] >= 0) {
        buffer[nP2P].alloc(
            yz_inner_slice,
            uint3{Gd        , Gd, Gd},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, yz_block_dim);
        buffer[nP2P + 1].alloc(
            yz_inner_slice,
            uint3{Gd - thick, Gd, Gd},
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[1], neighbour[1] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[1], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    uint3 xz_inner_slice{pdom.shape.x - Gdx2, thick, pdom.shape.z - Gdx2};
    dim3 xz_block_dim(8, 1, 8);
    if (neighbour[2] >= 0) {
        buffer[nP2P].alloc(
            xz_inner_slice,
            uint3{Gd, pdom.shape.y - Gd - thick, Gd},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, xz_block_dim);
        buffer[nP2P + 1].alloc(
            xz_inner_slice,
            uint3{Gd, pdom.shape.y - Gd       , Gd},
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[2], neighbour[2] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[2], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[3] >= 0) {
        buffer[nP2P].alloc(
            xz_inner_slice,
            uint3{Gd, Gd, Gd},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, xz_block_dim);
        buffer[nP2P + 1].alloc(
            xz_inner_slice,
            uint3{Gd, Gd - thick, Gd},
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[3], neighbour[3] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[3], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    uint3 xy_inner_slice{pdom.shape.x - Gdx2, pdom.shape.x - Gdx2, thick};
    dim3 xy_block_dim(8, 8, 1);
    if (neighbour[4] >= 0) {
        buffer[nP2P].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, pdom.shape.z - Gd - thick},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, xy_block_dim);
        buffer[nP2P + 1].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, pdom.shape.z - Gd        },
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[4], neighbour[4] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[4], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[5] >= 0) {
        buffer[nP2P].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, Gd},
            BufType::Out,
            buf_hdctype
        );
        CPML1dev_PackBuffer(buffer[nP2P], data, pdom, xy_block_dim);
        buffer[nP2P + 1].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, Gd - thick},
            BufType::In,
            buf_hdctype
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[5], neighbour[5] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[5], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
}

void CPM::CPML2dev_IExchange6ColoredFace(double *data, Mapper &pdom, unsigned int color, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, unsigned int buf_hdctype, MPI_Request *&req) {
    buffer = new CPMBuffer<double>[12];
    req    = new MPI_Request[12];
    nP2P   = 0;
    uint3 yz_inner_slice{thick, pdom.shape.y - Gdx2, pdom.shape.z - Gdx2};
    dim3 yz_block_dim(1, 8, 8);
    if (neighbour[0] >= 0) {
        buffer[nP2P].alloc(
            yz_inner_slice,
            uint3{pdom.shape.x - Gd - thick, Gd, Gd},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, yz_block_dim);
        buffer[nP2P + 1].alloc(
            yz_inner_slice,
            uint3{pdom.shape.x - Gd        , Gd, Gd},
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[0], neighbour[0] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[0], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[1] >= 0) {
        buffer[nP2P].alloc(
            yz_inner_slice,
            uint3{Gd        , Gd, Gd},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, yz_block_dim);
        buffer[nP2P + 1].alloc(
            yz_inner_slice,
            uint3{Gd - thick, Gd, Gd},
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[1], neighbour[1] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[1], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    uint3 xz_inner_slice{pdom.shape.x - Gdx2, thick, pdom.shape.z - Gdx2};
    dim3 xz_block_dim(8, 1, 8);
    if (neighbour[2] >= 0) {
        buffer[nP2P].alloc(
            xz_inner_slice,
            uint3{Gd, pdom.shape.y - Gd - thick, Gd},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, xz_block_dim);
        buffer[nP2P + 1].alloc(
            xz_inner_slice,
            uint3{Gd, pdom.shape.y - Gd       , Gd},
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[2], neighbour[2] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[2], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[3] >= 0) {
        buffer[nP2P].alloc(
            xz_inner_slice,
            uint3{Gd, Gd, Gd},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, xz_block_dim);
        buffer[nP2P + 1].alloc(
            xz_inner_slice,
            uint3{Gd, Gd - thick, Gd},
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[3], neighbour[3] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[3], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    uint3 xy_inner_slice{pdom.shape.x - Gdx2, pdom.shape.x - Gdx2, thick};
    dim3 xy_block_dim(8, 8, 1);
    if (neighbour[4] >= 0) {
        buffer[nP2P].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, pdom.shape.z - Gd - thick},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, xy_block_dim);
        buffer[nP2P + 1].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, pdom.shape.z - Gd        },
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[4], neighbour[4] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[4], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
    if (neighbour[5] >= 0) {
        buffer[nP2P].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, Gd},
            BufType::Out,
            buf_hdctype,
            pdom, color
        );
        CPML1dev_PackColoredBuffer(buffer[nP2P], data, pdom, xy_block_dim);
        buffer[nP2P + 1].alloc(
            xy_inner_slice,
            uint3{Gd, Gd, Gd - thick},
            BufType::In,
            buf_hdctype,
            pdom, color
        );
        CPML2_ISend(buffer[nP2P  ], neighbour[5], neighbour[5] + grp_tag, MPI_COMM_WORLD, &req[nP2P  ]);
        CPML2_IRecv(buffer[nP2P+1], neighbour[5], rank         + grp_tag, MPI_COMM_WORLD, &req[nP2P+1]);
        nP2P += 2;
    }
}

void CPM::CPML2dev_PostExchange6Face(double *data, Mapper &pdom, CPMBuffer<double> *&buffer, MPI_Request *&req) {
    dim3 yz_block_dim(1, 8, 8);
    dim3 xz_block_dim(8, 1, 8);
    dim3 xy_block_dim(8, 8, 1);
    int __nP2P = 0;
    if (neighbour[0] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, yz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[1] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, yz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[2] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, xz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[3] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, xz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[4] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, xy_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[5] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackBuffer(buffer[__nP2P+1], data, pdom, xy_block_dim);
    }
    nP2P = 0;
    delete[] buffer;
    delete[] req;
}

void CPM::CPML2dev_PostExchange6ColoredFace(double *data, Mapper &pdom, unsigned int color, CPMBuffer<double> *&buffer, MPI_Request *&req) {
    dim3 yz_block_dim(1, 8, 8);
    dim3 xz_block_dim(8, 1, 8);
    dim3 xy_block_dim(8, 8, 1);
    int __nP2P = 0;
    if (neighbour[0] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, yz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[1] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, yz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[2] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, xz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[3] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, xz_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[4] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, xy_block_dim);
        buffer[__nP2P+1].release();
        __nP2P += 2;
    }
    if (neighbour[5] >= 0) {
        buffer[__nP2P].release();
        CPML1dev_UnpackColoredBuffer(buffer[__nP2P+1], data, pdom, xy_block_dim);
    }
    nP2P = 0;
    delete[] buffer;
    delete[] req;
}

}
