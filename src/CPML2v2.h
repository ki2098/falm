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
    return MPI_Isend(buffer.ptr, buffer.count, mpi_dtype, dst, tag, mpi_comm, mpi_req);
}

static inline int CPML2_IRecv(CPMBuffer &buffer, MPI_Datatype mpi_dtype, int src, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Irecv(buffer.ptr, buffer.count, mpi_dtype, src, tag, mpi_comm, mpi_req);
}

static inline int CPML2_Wait(MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Wait(mpi_req, mpi_status);
}

static inline int CPML2_Waitall(int count, MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Waitall(count, mpi_req, mpi_status);
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

static inline int CPML2_AllReduce(void *buffer, int count, MPI_Datatype mpi_dtype, MPI_Op mpi_op, MPI_Comm mpi_comm) {
    return MPI_Allreduce(MPI_IN_PLACE, buffer, count, mpi_dtype, mpi_op, mpi_comm);
}

class CPMBase {
public:
    int neighbour[6];
    INTx3   shape;
    INTx3     idx;
    int      rank;
    int      size;
    bool use_cuda_aware_mpi;
    CPMBase(bool _use_cuda_aware_mpi = false) : use_cuda_aware_mpi(_use_cuda_aware_mpi) {}
    CPMBase(const CPMBase &_cpmbase, bool _use_cuda_aware_mpi = false) : shape(_cpmbase.shape), idx(_cpmbase.idx), rank(_cpmbase.rank), size(_cpmbase.size), use_cuda_aware_mpi(_use_cuda_aware_mpi) {
        memcpy(neighbour, _cpmbase.neighbour, sizeof(int) * 6);
    }
    void initNeighbour() {
        int __rank = rank;
        INT i, j, k;
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

    void setRegions(INTx3 &inner_shape, INTx3 &inner_offset, INTx3 *boundary_shape, INTx3 *boundary_offset, INT thick, Mapper &pdm) {
        inner_shape = {
            pdm.shape.x - Gdx2,
            pdm.shape.y - Gdx2,
            pdm.shape.z - Gdx2
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

    bool validNeighbour(INT fid) {
        return (neighbour[fid] >= 0);
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
    T              *origin_ptr;
    Mapper        origin_domain;
    FLAG         buffer_hdctype;
    void             *packerptr[6];
    

    CPMOp() : origin_ptr(nullptr) {}
    CPMOp(const CPMBase &_base) : 
        base(_base, _base.use_cuda_aware_mpi),
        origin_ptr(nullptr)
    {
        mpi_dtype = getMPIDtype<T>();
        if (base.use_cuda_aware_mpi) {
            buffer_hdctype = HDCType::Device;
        } else {
            buffer_hdctype = HDCType::Host;
        }
    }
    CPMOp(const CPMBase &_base, bool _use_cuda_aware_mpi) :
        base(_base, _use_cuda_aware_mpi),
        origin_ptr(nullptr)
    {
        mpi_dtype = getMPIDtype<T>();
        if (base.use_cuda_aware_mpi) {
            buffer_hdctype = HDCType::Device;
        } else {
            buffer_hdctype = HDCType::Host;
        }
    }

    void CPML2_Wait6Face() {
        for (INT i = 0; i < 6; i ++) {
            if (base.neighbour[i] >= 0) {
                CPML2_Waitall(2, &mpi_req[i * 2], &mpi_stat[i * 2]);
            }
        }
    }

    // void CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag);
    // void CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag);
    // void CPML2Dev_PostExchange6Face();
    // void CPML2Dev_PostExchange6ColoredFace();
    void CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag, STREAM *stream = nullptr);
    void CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag, STREAM *stream = nullptr);
    void CPML2Dev_PostExchange6Face(STREAM *stream = nullptr);
    void CPML2Dev_PostExchange6ColoredFace(STREAM *stream = nullptr);

protected:
    void makeBufferShapeOffset(Mapper &pdm, INT thick) {
        for (INT fid = 0; fid < 6; fid ++) {
            if (base.neighbour[fid] >= 0) {
                INT __s = fid*2, __r = fid*2+1;
                INTx3 buffer_shape {
                    (fid / 2 == 0)? thick : pdm.shape.x - Gdx2,
                    (fid / 2 == 1)? thick : pdm.shape.y - Gdx2,
                    (fid / 2 == 2)? thick : pdm.shape.z - Gdx2
                };
                INTx3 sendbuffer_offset, recvbuffer_offset;
                if (fid == 0) {
                    sendbuffer_offset = {pdm.shape.x - Gd - thick, Gd, Gd};
                    recvbuffer_offset = {pdm.shape.x - Gd        , Gd, Gd};
                } else if (fid == 1) {
                    sendbuffer_offset = {               Gd        , Gd, Gd};
                    recvbuffer_offset = {               Gd - thick, Gd, Gd};
                } else if (fid == 2) {
                    sendbuffer_offset = {Gd, pdm.shape.y - Gd - thick, Gd};
                    recvbuffer_offset = {Gd, pdm.shape.y - Gd        , Gd};
                } else if (fid == 3) {
                    sendbuffer_offset = {Gd,                Gd        , Gd};
                    recvbuffer_offset = {Gd,                Gd - thick, Gd};
                } else if (fid == 4) {
                    sendbuffer_offset = {Gd, Gd, pdm.shape.z - Gd - thick};
                    recvbuffer_offset = {Gd, Gd, pdm.shape.z - Gd        };
                } else if (fid == 5) {
                    sendbuffer_offset = {Gd, Gd,                Gd        };
                    recvbuffer_offset = {Gd, Gd,                Gd - thick};
                }
                buffer[__s].map = Mapper(buffer_shape, sendbuffer_offset);
                buffer[__r].map = Mapper(buffer_shape, recvbuffer_offset);
            }
        }
    }

};

template<typename T> void CPMOp<T>::CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag, STREAM *stream) {
    assert(origin_ptr == nullptr);
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferShapeOffset(pdm, thick);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.alloc(sizeof(T), sbuf.map, BufType::Out, buffer_hdctype);
            rbuf.alloc(sizeof(T), rbuf.map, BufType::In , buffer_hdctype);
            if (buffer_hdctype == HDCType::Host) {
                packerptr[fid] = falmMallocDevice(sizeof(T) * sbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            CPMBuffer &sbuf = buffer[fid * 2];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPML0Dev_PackBuffer((T*)sbuf.ptr, sbuf.map, data, pdm, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPML0Dev_PackBuffer((T*)packerptr[fid], sbuf.map, data, pdm, block_dim, fstream);
                falmMemcpyAsync(sbuf.ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            if (buffer_hdctype == HDCType::Host) {
                falmFreeDevice(packerptr[fid]);
            }
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPML2_ISend(sbuf, mpi_dtype, base.neighbour[fid], base.neighbour[fid] + grp_tag, MPI_COMM_WORLD, &sreq);
            CPML2_IRecv(rbuf, mpi_dtype, base.neighbour[fid], base.rank           + grp_tag, MPI_COMM_WORLD, &rreq);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag, STREAM *stream) {
    assert(origin_ptr == nullptr);
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferShapeOffset(pdm, thick);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.allocColored(sizeof(T), sbuf.map, color, BufType::Out, buffer_hdctype, pdm);
            rbuf.allocColored(sizeof(T), rbuf.map, color, BufType::In , buffer_hdctype, pdm);
            if (buffer_hdctype == HDCType::Host) {
                packerptr[fid] = falmMallocDevice(sizeof(T) * sbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            CPMBuffer &sbuf = buffer[fid * 2];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPML0Dev_PackColoredBuffer((T*)sbuf.ptr, sbuf.map, color, data, pdm, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPML0Dev_PackColoredBuffer((T*)packerptr[fid], sbuf.map, color, data, pdm, block_dim, fstream);
                falmMemcpyAsync(sbuf.ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            if (buffer_hdctype == HDCType::Host) {
                falmFree(packerptr[fid]);
            }
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPML2_ISend(sbuf, mpi_dtype, base.neighbour[fid], base.neighbour[fid] + grp_tag, MPI_COMM_WORLD, &sreq);
            CPML2_IRecv(rbuf, mpi_dtype, base.neighbour[fid], base.rank           + grp_tag, MPI_COMM_WORLD, &rreq);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2Dev_PostExchange6Face(STREAM *stream) {
    assert(origin_ptr != nullptr);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                packerptr[fid] = falmMallocDevice(sizeof(T) * rbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            CPMBuffer &rbuf = buffer[fid * 2 + 1];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPML0Dev_UnpackBuffer((T*)rbuf.ptr, rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                falmMemcpyAsync(packerptr[fid], rbuf.ptr, sizeof(T) * rbuf.count, MCpType::Hst2Dev, fstream);
                CPML0Dev_UnpackBuffer((T*)packerptr[fid], rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            if (buffer_hdctype == HDCType::Host) {
                falmFreeDevice(packerptr[fid]);
            }
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.release();
            rbuf.release();
        }
    }
    origin_ptr = nullptr;
}

template<typename T> void CPMOp<T>::CPML2Dev_PostExchange6ColoredFace(STREAM *stream) {
    assert(origin_ptr != nullptr);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                packerptr[fid] = falmMallocDevice(sizeof(T) * rbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            dim3 block_dim(
                (fid / 2 == 0)? 1U : 8U,
                (fid / 2 == 1)? 1U : 8U,
                (fid / 2 == 2)? 1U : 8U
            );
            CPMBuffer &rbuf = buffer[fid * 2 + 1];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPML0Dev_UnpackColoredBuffer((T*)rbuf.ptr, rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                falmMemcpyAsync(packerptr[fid], rbuf.ptr, sizeof(T) * rbuf.count, MCpType::Hst2Dev, fstream);
                CPML0Dev_UnpackColoredBuffer((T*)packerptr[fid], rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base.validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            if (buffer_hdctype == HDCType::Host) {
                falmFreeDevice(packerptr[fid]);
            }
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.release();
            rbuf.release();
        }
    }
    origin_ptr = nullptr;
}

// template<typename T> void CPMOp<T>::CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag) {
//     assert(oringin_ptr == nullptr);
//     oringin_ptr = data;
//     oringin_data_domain = pdm;
//     int *neighbour = base.neighbour;
//     for (INT i = 0; i < 6; i ++) {
//         if (neighbour[i] >= 0) {
//             dim3 block_dim(
//                 (i / 2 == 0)? 1U : 8U,
//                 (i / 2 == 1)? 1U : 8U,
//                 (i / 2 == 2)? 1U : 8U
//             );
//             INTx3 buffer_shape {
//                 (i / 2 == 0)? thick : pdm.shape.x - Gdx2,
//                 (i / 2 == 1)? thick : pdm.shape.y - Gdx2,
//                 (i / 2 == 2)? thick : pdm.shape.z - Gdx2
//             };
//             INTx3 sendbuffer_offset, recvbuffer_offset;
//             if (i == 0) {
//                 sendbuffer_offset = {pdm.shape.x - Gd - thick, Gd, Gd};
//                 recvbuffer_offset = {pdm.shape.x - Gd        , Gd, Gd};
//             } else if (i == 1) {
//                 sendbuffer_offset = {               Gd        , Gd, Gd};
//                 recvbuffer_offset = {               Gd - thick, Gd, Gd};
//             } else if (i == 2) {
//                 sendbuffer_offset = {Gd, pdm.shape.y - Gd - thick, Gd};
//                 recvbuffer_offset = {Gd, pdm.shape.y - Gd        , Gd};
//             } else if (i == 3) {
//                 sendbuffer_offset = {Gd,                Gd        , Gd};
//                 recvbuffer_offset = {Gd,                Gd - thick, Gd};
//             } else if (i == 4) {
//                 sendbuffer_offset = {Gd, Gd, pdm.shape.z - Gd - thick};
//                 recvbuffer_offset = {Gd, Gd, pdm.shape.z - Gd        };
//             } else if (i == 5) {
//                 sendbuffer_offset = {Gd, Gd,                Gd        };
//                 recvbuffer_offset = {Gd, Gd,                Gd - thick};
//             }
//             buffer[i*2].alloc(
//                 sizeof(T),
//                 Mapper(buffer_shape, sendbuffer_offset),
//                 BufType::Out,
//                 buffer_hdctype
//             );
//             buffer[i*2+1].alloc(
//                 sizeof(T),
//                 Mapper(buffer_shape, recvbuffer_offset),
//                 BufType::In,
//                 buffer_hdctype
//             );
//             CPML1Dev_PackBuffer(buffer[i*2], data, pdm, block_dim);
//             CPML2_ISend(buffer[i*2  ], mpi_dtype, neighbour[i], neighbour[i] + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2  ]);
//             CPML2_IRecv(buffer[i*2+1], mpi_dtype, neighbour[i], base.rank    + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2+1]);
//         }
//     }
// }

// template<typename T> void CPMOp<T>::CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag) {
//     assert(oringin_ptr == nullptr);
//     oringin_ptr = data;
//     oringin_data_domain = pdm;
//     int *neighbour = base.neighbour;
//     for (INT i = 0; i < 6; i ++) {
//         if (neighbour[i] >= 0) {
//             dim3 block_dim(
//                 (i / 2 == 0)? 1U : 8U,
//                 (i / 2 == 1)? 1U : 8U,
//                 (i / 2 == 2)? 1U : 8U
//             );
//             INTx3 buffer_shape {
//                 (i / 2 == 0)? thick : pdm.shape.x - Gdx2,
//                 (i / 2 == 1)? thick : pdm.shape.y - Gdx2,
//                 (i / 2 == 2)? thick : pdm.shape.z - Gdx2
//             };
//             INTx3 sendbuffer_offset, recvbuffer_offset;
//             if (i == 0) {
//                 sendbuffer_offset = {pdm.shape.x - Gd - thick, Gd, Gd};
//                 recvbuffer_offset = {pdm.shape.x - Gd        , Gd, Gd};
//             } else if (i == 1) {
//                 sendbuffer_offset = {               Gd        , Gd, Gd};
//                 recvbuffer_offset = {               Gd - thick, Gd, Gd};
//             } else if (i == 2) {
//                 sendbuffer_offset = {Gd, pdm.shape.y - Gd - thick, Gd};
//                 recvbuffer_offset = {Gd, pdm.shape.y - Gd        , Gd};
//             } else if (i == 3) {
//                 sendbuffer_offset = {Gd,                Gd        , Gd};
//                 recvbuffer_offset = {Gd,                Gd - thick, Gd};
//             } else if (i == 4) {
//                 sendbuffer_offset = {Gd, Gd, pdm.shape.z - Gd - thick};
//                 recvbuffer_offset = {Gd, Gd, pdm.shape.z - Gd        };
//             } else if (i == 5) {
//                 sendbuffer_offset = {Gd, Gd,                Gd        };
//                 recvbuffer_offset = {Gd, Gd,                Gd - thick};
//             }
//             buffer[i*2].allocColored(
//                 sizeof(T),
//                 Mapper(buffer_shape, sendbuffer_offset),
//                 BufType::Out,
//                 buffer_hdctype,
//                 pdm, color
//             );
//             buffer[i*2+1].allocColored(
//                 sizeof(T),
//                 Mapper(buffer_shape, recvbuffer_offset),
//                 recvbuffer_offset,
//                 BufType::In,
//                 buffer_hdctype,
//                 pdm, color
//             );
//             CPML1Dev_PackColoredBuffer(buffer[i*2], data, pdm, block_dim);
//             CPML2_ISend(buffer[i*2  ], mpi_dtype, neighbour[i], neighbour[i] + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2  ]);
//             CPML2_IRecv(buffer[i*2+1], mpi_dtype, neighbour[i], base.rank    + grp_tag, MPI_COMM_WORLD, &mpi_req[i*2+1]);
//         }
//     }
// }

// template<typename T> void CPMOp<T>::CPML2Dev_PostExchange6Face() {
//     assert(oringin_ptr != nullptr);
//     int *neighbour = base.neighbour;
//     for (INT i = 0; i < 6; i ++) {
//         if (neighbour[i] >= 0) {
//             dim3 block_dim(
//                 (i / 2 == 0)? 1U : 8U,
//                 (i / 2 == 1)? 1U : 8U,
//                 (i / 2 == 2)? 1U : 8U
//             );
//             buffer[i*2].release();
//             CPML1Dev_UnpackBuffer(buffer[i*2+1], oringin_ptr, oringin_data_domain, block_dim);
//             buffer[i*2+1].release();
//         }
//     }
//     oringin_ptr = nullptr;
// }

// template<typename T> void CPMOp<T>::CPML2Dev_PostExchange6ColoredFace() {
//     assert(oringin_ptr != nullptr);
//     int *neighbour = base.neighbour;
//     void *ptr[6];
//     for (INT i = 0; i < 6; i ++) {
//         if (neighbour[i] >= 0) {
//             dim3 block_dim(
//                 (i / 2 == 0)? 1U : 8U,
//                 (i / 2 == 1)? 1U : 8U,
//                 (i / 2 == 2)? 1U : 8U
//             );
//             buffer[i*2].release();
//             CPML1Dev_UnpackColoredBuffer(buffer[i*2+1], oringin_ptr, oringin_data_domain, block_dim);
//             buffer[i*2+1].release();
//         }
//     }
//     oringin_ptr = nullptr;
// }

}

#endif
