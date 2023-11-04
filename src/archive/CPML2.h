#ifndef FALM_CPML2_H
#define FALM_CPML2_H

#include <mpi.h>
#include "CPMB.h"
#include "CPML1.h"

namespace Falm {

static inline int CPML2_ISend(CPMBuffer<double> &buffer, int dst_rank, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Isend(buffer.ptr, buffer.size, MPI_DOUBLE, dst_rank, tag, mpi_comm, mpi_req);
}

static inline int CPML2_IRecv(CPMBuffer<double> &buffer, int src_rank, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Irecv(buffer.ptr, buffer.size, MPI_DOUBLE, src_rank, tag, mpi_comm, mpi_req);
}

static inline int CPM_Wait(MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Wait(mpi_req, mpi_status);
}

static inline int CPM_Waitall(int n, MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Waitall(n, mpi_req, mpi_status);
}

static inline int CPM_Init(int *argc, char ***argv) {
    return MPI_Init(argc, argv);
}

static inline int CPM_Finalize() {
    return MPI_Finalize();
}

static inline int CPM_GetRank(MPI_Comm mpi_comm, int &mpi_rank) {
    return MPI_Comm_rank(mpi_comm, &mpi_rank);
}

static inline int CPM_GetSize(MPI_Comm mpi_comm, int &mpi_size) {
    return MPI_Comm_size(mpi_comm, &mpi_size);
}

static inline int CPM_Barrier(MPI_Comm mpi_comm) {
    return MPI_Barrier(mpi_comm);
}

static inline int CPM_AllReduce(void *buffer, int n, MPI_Datatype mpi_dtype, MPI_Op mpi_op, MPI_Comm mpi_comm) {
    return MPI_Allreduce(MPI_IN_PLACE, buffer, n, mpi_dtype, mpi_op, mpi_comm);
}

class CPM {
public:
    int neighbour[6];
    uint3   shape;
    uint3     idx;
    int      rank;
    int      size;
    int      nP2P;
    CPM() {}
    CPM(const CPM &cpm) : shape(cpm.shape), idx(cpm.idx), rank(cpm.rank), size(cpm.size) {
        memcpy(neighbour, cpm.neighbour, sizeof(int) * 6);
    }
    void init_neighbour() {
        int __rank = rank;
        int i, j, k;
        k = __rank / (shape[0] * shape[1]);
        __rank = __rank % (shape[0] * shape[1]);
        j = __rank / shape[0];
        i = __rank % shape[0];
        idx[0] = i;
        idx[1] = j;
        idx[2] = k;
        neighbour[0] = IDX(i + 1, j, k, shape);
        neighbour[1] = IDX(i - 1, j, k, shape);
        neighbour[2] = IDX(i, j + 1, k, shape);
        neighbour[3] = IDX(i, j - 1, k, shape);
        neighbour[4] = IDX(i, j, k + 1, shape);
        neighbour[5] = IDX(i, j, k - 1, shape);
        if (i == shape[0] - 1) {
            neighbour[0] = - 1;
        }
        if (i == 0) {
            neighbour[1] = - 1;
        }
        if (j == shape[1] - 1) {
            neighbour[2] = - 1;
        }
        if (j == 0) {
            neighbour[3] = - 1;
        }
        if (k == shape[2] - 1) {
            neighbour[4] = - 1;
        }
        if (k == 0) {
            neighbour[5] = - 1;
        }
    }

    void CPML2dev_IExchange6Face(double *data, Region &pdm, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, unsigned int buf_hdctype, MPI_Request *&req);
    void CPML2dev_IExchange6ColoredFace(double *data, Region &pdm, unsigned int color, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, unsigned int buf_hdctype, MPI_Request *&req);
    void CPML2dev_PostExchange6Face(double *data, Region &pdm, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void CPML2dev_PostExchange6ColoredFace(double *data, Region &pdm, unsigned int color, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void CPML2_Wait6Face(MPI_Request *req);

    void setRegions(uint3 &inner_shape, uint3 &inner_offset, uint3 *boundary_shape, uint3 *boundary_offset, unsigned int thick, Region &pdm) {
        inner_shape = {
            pdm.shape[0] - Gdx2,
            pdm.shape[1] - Gdx2,
            pdm.shape[2] - Gdx2
        };
        inner_offset = {Gd, Gd, Gd};
        if (neighbour[0] >= 0) {
            boundary_shape[0]  = {thick, inner_shape[1], inner_shape[2]};
            boundary_offset[0] = {inner_offset[0] + inner_shape[0] - thick, inner_offset[1], inner_offset[2]};
            inner_shape[0] -= thick;
        }
        if (neighbour[1] >= 0) {
            boundary_shape[1]  = {thick, inner_shape[1], inner_shape[2]};
            boundary_offset[1] = {inner_offset[0], inner_offset[1], inner_offset[2]};
            inner_shape[0]  -= thick;
            inner_offset[0] += thick; 
        }
        if (neighbour[2] >= 0) {
            boundary_shape[2]  = {inner_shape[0], thick, inner_shape[2]};
            boundary_offset[2] = {inner_offset[0], inner_offset[1] + inner_shape[1] - thick, inner_offset[2]};
            inner_shape[1] -= thick;
        }
        if (neighbour[3] >= 0) {
            boundary_shape[3]  = {inner_shape[0], thick, inner_shape[2]};
            boundary_offset[3] = {inner_offset[0], inner_offset[1], inner_offset[2]};
            inner_shape[1]  -= thick;
            inner_offset[1] += thick;
        }
        if (neighbour[4] >= 0) {
            boundary_shape[4]  = {inner_shape[0], inner_shape[1], thick};
            boundary_offset[4] = {inner_offset[0], inner_offset[1], inner_offset[2] + inner_shape[2] - thick};
            inner_shape[2] -= thick;
        }
        if (neighbour[5] >= 0) {
            boundary_shape[5]  = {inner_shape[0], inner_shape[1], thick};
            boundary_offset[5] = {inner_offset[0], inner_offset[1], inner_offset[2]};
            inner_shape[2]  -= thick;
            inner_offset[2] += thick;
        }
    }
};

}

#endif
