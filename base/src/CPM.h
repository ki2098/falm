#ifndef FALM_CPM_H
#define FALM_CPM_H

#include <mpi.h>
#include "CPMB.h"

namespace Falm {

static inline int CPM_ISend(CPMBuffer<double> &buffer, int dst_rank, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Isend(buffer.ptr, buffer.size, MPI_DOUBLE, dst_rank, tag, mpi_comm, mpi_req);
}

static inline int CPM_IRecv(CPMBuffer<double> &buffer, int src_rank, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
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
        k = __rank / (shape.x * shape.y);
        __rank = __rank % (shape.x * shape.y);
        j = __rank / shape.x;
        i = __rank % shape.x;
        idx.x = i;
        idx.y = j;
        idx.z = k;
        neighbour[0] = IDX(i + 1, j, k, shape);
        neighbour[1] = IDX(i - 1, j, k, shape);
        neighbour[2] = IDX(i, j + 1, k, shape);
        neighbour[3] = IDX(i, j - 1, k, shape);
        neighbour[4] = IDX(i, j, k + 1, shape);
        neighbour[5] = IDX(i, j, k - 1, shape);
        if (i == shape.x - 1) {
            neighbour[0] = - 1;
        }
        if (i == 0) {
            neighbour[1] = - 1;
        }
        if (j == shape.y - 1) {
            neighbour[2] = - 1;
        }
        if (j == 0) {
            neighbour[3] = - 1;
        }
        if (k == shape.z - 1) {
            neighbour[4] = - 1;
        }
        if (k == 0) {
            neighbour[5] = - 1;
        }
    }

    void dev_IExchange6Face(double *data, Mapper &pdom, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void dev_IExchange6ColoredFace(double *data, Mapper &pdom, unsigned int color, unsigned int thick, int grp_tag, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void dev_PostExchange6Face(double *data, Mapper &pdom, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void dev_PostExchange6ColoredFace(double *data, Mapper &pdom, unsigned int color, CPMBuffer<double> *&buffer, MPI_Request *&req);
    void Wait6Face(MPI_Request *req);
};

}

#endif