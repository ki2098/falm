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

}

#endif