#ifndef FALM_CPML2V2_H
#define FALM_CPML2V2_H

#include <mpi.h>
#include <typeinfo>
#include "CPMBv2.h"
#include "CPML1v2.h"

namespace Falm {

template<typename T> MPI_Datatype getMPIDtype() {
    if     (typeid(T) == typeid(char))               return MPI_CHAR;
    else if(typeid(T) == typeid(short))              return MPI_SHORT;
    else if(typeid(T) == typeid(int))                return MPI_INT;
    else if(typeid(T) == typeid(long))               return MPI_LONG;
    else if(typeid(T) == typeid(float))              return MPI_FLOAT;
    else if(typeid(T) == typeid(double))             return MPI_DOUBLE;
    else if(typeid(T) == typeid(long double))        return MPI_LONG_DOUBLE;
    else if(typeid(T) == typeid(unsigned char))      return MPI_UNSIGNED_CHAR;
    else if(typeid(T) == typeid(unsigned short))     return MPI_UNSIGNED_SHORT;
    else if(typeid(T) == typeid(unsigned))           return MPI_UNSIGNED;
    else if(typeid(T) == typeid(unsigned int))       return MPI_UNSIGNED;
    else if(typeid(T) == typeid(unsigned long))      return MPI_UNSIGNED_LONG;
#ifdef MPI_LONG_LONG_INT
    else if(typeid(T) == typeid(long long int))      return MPI_LONG_LONG_INT;
#endif
#ifdef MPI_LONG_LONG
    else if(typeid(T) == typeid(long long))          return MPI_LONG_LONG;
#endif
#ifdef MPI_UNSIGNED_LONG_LONG
    else if(typeid(T) == typeid(unsigned long long)) return MPI_UNSIGNED_LONG_LONG;
#endif
    return MPI_DATATYPE_NULL;
}

static inline int CPML2_ISend(CPMBuffer &buffer, MPI_Datatype mpi_dtype, int dst, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Isend(buffer.ptr, buffer.size, mpi_dtype, dst, tag, mpi_comm, mpi_req);
}

static inline int CPML2_IRecv(CPMBuffer &buffer, MPI_Datatype mpi_dtype, int src, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Irecv(buffer.ptr, buffer.size, mpi_dtype, src, tag, mpi_comm, mpi_req);
}

static inline int CPML2_Wait(MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Wait(mpi_req, mpi_status);
}

static inline int CPML2_Waitall(int n, MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Waitall(n, mpi_req, mpi_status);
}

static inline int CPML2_Init(int *argc, char ***argv) {
    return MPI_Init(argc, argv);
}

static inline int CPML2_Finalize() {
    return MPI_Finalize();
}

static inline int CPML2_GetRank(MPI_Comm mpi_comm, int &mpi_rank) {
    return MPI_Comm_rank(mpi_comm, &mpi_rank);
}

static inline int CPML2_GetSize(MPI_Comm mpi_comm, int &mpi_size) {
    return MPI_Comm_size(mpi_comm, &mpi_size);
}

static inline int CPML2_Barrier(MPI_Comm mpi_comm) {
    return MPI_Barrier(mpi_comm);
}

static inline int CPML2_AllReduce(void *buffer, int n, MPI_Datatype mpi_dtype, MPI_Op mpi_op, MPI_Comm mpi_comm) {
    return MPI_Allreduce(MPI_IN_PLACE, buffer, n, mpi_dtype, mpi_op, mpi_comm);
}

class CPMBase {
public:
    int            neighbour[6];
    uint3             shape;
    uint3               idx;
    int                rank;
    int                size;
    bool use_cuda_aware_mpi;
    CPMBase(bool _use_cuda_aware_mpi = false) : use_cuda_aware_mpi(_use_cuda_aware_mpi) {}
    CPMBase(const CPMBase &_cpmbase, bool _use_cuda_aware_mpi = false) : shape(_cpmbase.shape), idx(_cpmbase.idx), rank(_cpmbase.rank), size(_cpmbase.size), use_cuda_aware_mpi(_use_cuda_aware_mpi) {
        memcpy(neighbour, _cpmbase.neighbour, sizeof(int) * 6);
    }
    void initNeighbour() {
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

    void setRegions(uint3 &inner_shape, uint3 &inner_offset, uint3 *boundary_shape, uint3 *boundary_offset, unsigned int thick, Mapper &pdom) {
        inner_shape = {
            pdom.shape.x - Gdx2,
            pdom.shape.y - Gdx2,
            pdom.shape.z - Gdx2
        };
        inner_offset = {Gd, Gd, Gd};
        if (neighbour[0] >= 0) {
            boundary_shape[0]  = {thick, inner_shape.y, inner_shape.z};
            boundary_offset[0] = {inner_offset.x + inner_shape.x - thick, inner_offset.y, inner_offset.z};
            inner_shape.x -= thick;
        }
        if (neighbour[1] >= 0) {
            boundary_shape[1]  = {thick, inner_shape.y, inner_shape.z};
            boundary_offset[1] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.x  -= thick;
            inner_offset.x += thick; 
        }
        if (neighbour[2] >= 0) {
            boundary_shape[2]  = {inner_shape.x, thick, inner_shape.z};
            boundary_offset[2] = {inner_offset.x, inner_offset.y + inner_shape.y - thick, inner_offset.z};
            inner_shape.y -= thick;
        }
        if (neighbour[3] >= 0) {
            boundary_shape[3]  = {inner_shape.x, thick, inner_shape.z};
            boundary_offset[3] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.y  -= thick;
            inner_offset.y += thick;
        }
        if (neighbour[4] >= 0) {
            boundary_shape[4]  = {inner_shape.x, inner_shape.y, thick};
            boundary_offset[4] = {inner_offset.x, inner_offset.y, inner_offset.z + inner_shape.z - thick};
            inner_shape.z -= thick;
        }
        if (neighbour[5] >= 0) {
            boundary_shape[5]  = {inner_shape.x, inner_shape.y, thick};
            boundary_offset[5] = {inner_offset.x, inner_offset.y, inner_offset.z};
            inner_shape.z  -= thick;
            inner_offset.z += thick;
        }
    }
};



template<typename T>
class CPMOp {
public:
    CPMBase                base;
    CPMBuffer            buffer[12];
    MPI_Request         mpi_req[12];
    MPI_Status         mpi_stat[12];
    MPI_Datatype      mpi_dtype;
    T              *oringin_ptr;
    Mapper  oringin_data_domain;
    unsigned int buffer_hdctype;
    

    CPMOp() : oringin_ptr(nullptr) {}
    CPMOp(const CPMBase &_base) : 
        base(_base, _base.use_cuda_aware_mpi),
        oringin_ptr(nullptr)
    {
        mpi_dtype = getMPIDtype<T>();
        if (base.use_cuda_aware_mpi) {
            buffer_hdctype = HDCType::Device;
        } else {
            buffer_hdctype = HDCType::Host;
        }
    }

    void CPML2_Wait6Face() {
        for (int i = 0; i < 6; i ++) {
            if (base.neighbour[i] >= 0) {
                CPML2_Waitall(2, &mpi_req[i * 2], &mpi_stat[i * 2]);
            }
        }
    }

    void CPML2dev_IExchange6Face(T *data, Mapper &pdom, unsigned int thick, int grp_tag);
    void CPML2dev_IExchange6ColoredFace(T *data, Mapper &pdom, unsigned int color, unsigned int thick, int grp_tag);
    void CPML2dev_PostExchange6Face();
    void CPML2dev_PostExchange6ColoredFace();
};

template<typename T> void CPMOp<T>::CPML2dev_IExchange6Face(T *data, Mapper &pdom, unsigned int thick, int grp_tag) {
    assert(oringin_ptr == nullptr);
    oringin_ptr = data;
    oringin_data_domain = pdom;
    int *neighbour = base.neighbour;
    for (int i = 0; i < 6; i ++) {
        if (neighbour[i] >= 0) {
            dim3 block_dim(
                (i / 2 == 0)? 1U : 8U,
                (i / 2 == 1)? 1U : 8U,
                (i / 2 == 2)? 1U : 8U
            );
            uint3 buffer_shape {
                (i / 2 == 0)? thick : pdom.shape.x - Gdx2,
                (i / 2 == 1)? thick : pdom.shape.y - Gdx2,
                (i / 2 == 2)? thick : pdom.shape.z - Gdx2
            };
            uint3 sendbuffer_offset, recvbuffer_offset;
            if (i == 0) {
                sendbuffer_offset = {pdom.shape.x - Gd - thick, Gd, Gd};
                recvbuffer_offset = {pdom.shape.x - Gd        , Gd, Gd};
            } else if (i == 1) {
                sendbuffer_offset = {               Gd        , Gd, Gd};
                recvbuffer_offset = {               Gd - thick, Gd, Gd};
            } else if (i == 2) {
                sendbuffer_offset = {Gd, pdom.shape.y - Gd - thick, Gd};
                recvbuffer_offset = {Gd, pdom.shape.y - Gd        , Gd};
            } else if (i == 3) {
                sendbuffer_offset = {Gd,                Gd        , Gd};
                recvbuffer_offset = {Gd,                Gd - thick, Gd};
            } else if (i == 4) {
                sendbuffer_offset = {Gd, Gd, pdom.shape.z - Gd - thick};
                recvbuffer_offset = {Gd, Gd, pdom.shape.z - Gd        };
            } else if (i == 5) {
                sendbuffer_offset = {Gd, Gd,                Gd        };
                recvbuffer_offset = {Gd, Gd,                Gd - thick};
            }
            buffer[i*2].alloc(
                sizeof(T),
                buffer_shape,
                sendbuffer_offset,
                BufType::Out,
                buffer_hdctype
            );
            buffer[i*2+1].alloc(
                sizeof(T),
                buffer_shape,
                recvbuffer_offset,
                BufType::In,
                buffer_hdctype
            );
            CPML1dev_PackBuffer(buffer[i*2], data, pdom, block_dim);
            CPML2_ISend(buffer[i*2  ], mpi_dtype, neighbour[i], neighbour[i] + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2  ]);
            CPML2_IRecv(buffer[i*2+1], mpi_dtype, neighbour[i], base.rank    + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2+1]);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2dev_IExchange6ColoredFace(T *data, Mapper &pdom, unsigned int color, unsigned int thick, int grp_tag) {
    assert(oringin_ptr == nullptr);
    oringin_ptr = data;
    oringin_data_domain = pdom;
    int *neighbour = base.neighbour;
    for (int i = 0; i < 6; i ++) {
        if (neighbour[i] >= 0) {
            dim3 block_dim(
                (i / 2 == 0)? 1U : 8U,
                (i / 2 == 1)? 1U : 8U,
                (i / 2 == 2)? 1U : 8U
            );
            uint3 buffer_shape {
                (i / 2 == 0)? thick : pdom.shape.x - Gdx2,
                (i / 2 == 1)? thick : pdom.shape.y - Gdx2,
                (i / 2 == 2)? thick : pdom.shape.z - Gdx2
            };
            uint3 sendbuffer_offset, recvbuffer_offset;
            if (i == 0) {
                sendbuffer_offset = {pdom.shape.x - Gd - thick, Gd, Gd};
                recvbuffer_offset = {pdom.shape.x - Gd        , Gd, Gd};
            } else if (i == 1) {
                sendbuffer_offset = {               Gd        , Gd, Gd};
                recvbuffer_offset = {               Gd - thick, Gd, Gd};
            } else if (i == 2) {
                sendbuffer_offset = {Gd, pdom.shape.y - Gd - thick, Gd};
                recvbuffer_offset = {Gd, pdom.shape.y - Gd        , Gd};
            } else if (i == 3) {
                sendbuffer_offset = {Gd,                Gd        , Gd};
                recvbuffer_offset = {Gd,                Gd - thick, Gd};
            } else if (i == 4) {
                sendbuffer_offset = {Gd, Gd, pdom.shape.z - Gd - thick};
                recvbuffer_offset = {Gd, Gd, pdom.shape.z - Gd        };
            } else if (i == 5) {
                sendbuffer_offset = {Gd, Gd,                Gd        };
                recvbuffer_offset = {Gd, Gd,                Gd - thick};
            }
            buffer[i*2].alloc(
                sizeof(T),
                buffer_shape,
                sendbuffer_offset,
                BufType::Out,
                buffer_hdctype,
                pdom, color
            );
            buffer[i*2+1].alloc(
                sizeof(T),
                buffer_shape,
                recvbuffer_offset,
                BufType::In,
                buffer_hdctype,
                pdom, color
            );
            CPML1dev_PackColoredBuffer(buffer[i*2], data, pdom, block_dim);
            CPML2_ISend(buffer[i*2  ], mpi_dtype, neighbour[i], neighbour[i] + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2  ]);
            CPML2_IRecv(buffer[i*2+1], mpi_dtype, neighbour[i], base.rank    + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2+1]);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2dev_PostExchange6Face() {
    assert(oringin_ptr != nullptr);
    int *neighbour = base.neighbour;
    for (int i = 0; i < 6; i ++) {
        if (neighbour[i] >= 0) {
            dim3 block_dim(
                (i / 2 == 0)? 1U : 8U,
                (i / 2 == 1)? 1U : 8U,
                (i / 2 == 2)? 1U : 8U
            );
            buffer[i*2].release();
            CPML1dev_UnpackBuffer(buffer[i*2+1], oringin_ptr, oringin_data_domain, block_dim);
            buffer[i*2+1].release();
        }
    }
    oringin_ptr = nullptr;
}

template<typename T> void CPMOp<T>::CPML2dev_PostExchange6ColoredFace() {
    assert(oringin_ptr != nullptr);
    int *neighbour = base.neighbour;
    for (int i = 0; i < 6; i ++) {
        if (neighbour[i] >= 0) {
            dim3 block_dim(
                (i / 2 == 0)? 1U : 8U,
                (i / 2 == 1)? 1U : 8U,
                (i / 2 == 2)? 1U : 8U
            );
            buffer[i*2].release();
            CPML1dev_UnpackColoredBuffer(buffer[i*2+1], oringin_ptr, oringin_data_domain, block_dim);
            buffer[i*2+1].release();
        }
    }
    oringin_ptr = nullptr;
}

}

#endif