#ifndef FALM_CPML2V2_H
#define FALM_CPML2V2_H

#include <mpi.h>
#include <typeinfo>
#include "CPMDevCall.h"
#include "CPMBase.h"

namespace Falm {

template<typename T> inline MPI_Datatype getMPIDtype() {
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

static inline int CPM_ISend(CPMBuffer &buffer, MPI_Datatype mpi_dtype, int dst, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Isend(buffer.ptr, buffer.count, mpi_dtype, dst, tag, mpi_comm, mpi_req);
}

static inline int CPM_IRecv(CPMBuffer &buffer, MPI_Datatype mpi_dtype, int src, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Irecv(buffer.ptr, buffer.count, mpi_dtype, src, tag, mpi_comm, mpi_req);
}

static inline int CPM_Wait(MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Wait(mpi_req, mpi_status);
}

static inline int CPM_Waitall(int count, MPI_Request *mpi_req, MPI_Status *mpi_status) {
    return MPI_Waitall(count, mpi_req, mpi_status);
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

static inline int CPM_AllReduce(void *buffer, int count, MPI_Datatype mpi_dtype, MPI_Op mpi_op, MPI_Comm mpi_comm) {
    return MPI_Allreduce(MPI_IN_PLACE, buffer, count, mpi_dtype, mpi_op, mpi_comm);
}

template<typename T>
class CPMComm : public CPMDevCall {
public:
    CPM               *base;
    CPMBuffer            buffer[12];
    MPI_Request         mpi_req[12];
    MPI_Status         mpi_stat[12];
    MPI_Datatype      mpi_dtype;
    T              *origin_ptr;
    Region        origin_domain;
    FLAG         buffer_hdctype;
    void             *packerptr[6];
    

    CPMComm() : origin_ptr(nullptr) {}
    CPMComm(CPM *_base) : 
        base(_base),
        origin_ptr(nullptr)
    {
        mpi_dtype = getMPIDtype<T>();
        if (base->use_cuda_aware_mpi) {
            buffer_hdctype = HDCType::Device;
        } else {
            buffer_hdctype = HDCType::Host;
        }
    }

    void Wait6Face() {
        for (INT i = 0; i < 6; i ++) {
            if (base->neighbour[i] >= 0) {
                CPM_Waitall(2, &mpi_req[i * 2], &mpi_stat[i * 2]);
            }
        }
    }

    // void CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag);
    // void CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag);
    // void CPML2Dev_PostExchange6Face();
    // void CPML2Dev_PostExchange6ColoredFace();
    void IExchange6Face(T *data, INT thick, INT margin, int grp_tag, STREAM *stream = nullptr);
    void IExchange6ColoredFace(T *data, INT color, INT thick, INT margin, int grp_tag, STREAM *stream = nullptr);
    void PostExchange6Face(STREAM *stream = nullptr);
    void PostExchange6ColoredFace(STREAM *stream = nullptr);

protected:
    void makeBufferShapeOffset(INT thick, INT margin) {
        Region &pdm = base->pdm_list[base->rank];
        for (INT fid = 0; fid < 6; fid ++) {
            // printf("%d\n", base->gc);
            if (base->neighbour[fid] >= 0) {
                INT gc = base->gc;
                INT gcx2 = 2 * gc;
                INT __s = fid*2, __r = fid*2+1;
                INT3 buffer_shape {
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? thick : pdm.shape[0] - gcx2,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? thick : pdm.shape[1] - gcx2,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? thick : pdm.shape[2] - gcx2
                };
                INT3 sendbuffer_offset, recvbuffer_offset;
                if (fid == CPM::XPLUS) {
                    sendbuffer_offset = {{pdm.shape[0] - gc - thick - margin, gc, gc}};
                    recvbuffer_offset = {{pdm.shape[0] - gc         + margin, gc, gc}};
                } else if (fid == CPM::XMINUS) {
                    sendbuffer_offset = {{              gc         + margin, gc, gc}};
                    recvbuffer_offset = {{              gc - thick - margin, gc, gc}};
                } else if (fid == CPM::YPLUS) {
                    sendbuffer_offset = {{gc, pdm.shape[1] - gc - thick - margin, gc}};
                    recvbuffer_offset = {{gc, pdm.shape[1] - gc         + margin, gc}};
                } else if (fid == CPM::YMINUS) {
                    sendbuffer_offset = {{gc,               gc         + margin, gc}};
                    recvbuffer_offset = {{gc,               gc - thick - margin, gc}};
                } else if (fid == CPM::ZPLUS) {
                    sendbuffer_offset = {{gc, gc, pdm.shape[2] - gc - thick - margin}};
                    recvbuffer_offset = {{gc, gc, pdm.shape[2] - gc         + margin}};
                } else if (fid == CPM::ZMINUS) {
                    sendbuffer_offset = {{gc, gc,               gc         + margin}};
                    recvbuffer_offset = {{gc, gc,               gc - thick - margin}};
                }
                buffer[__s].map = Region(buffer_shape, sendbuffer_offset);
                buffer[__r].map = Region(buffer_shape, recvbuffer_offset);
            }
        }
    }

};

template<typename T> void CPMComm<T>::IExchange6Face(T *data, INT thick, INT margin, int grp_tag, STREAM *stream) {
    Region &pdm   = base->pdm_list[base->rank];
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferShapeOffset(thick, margin);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.alloc(sizeof(T), sbuf.map, BufType::Out, buffer_hdctype);
            rbuf.alloc(sizeof(T), rbuf.map, BufType::In , buffer_hdctype);
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * sbuf.count));
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CPMBuffer &sbuf = buffer[fid * 2];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPMDevCall::PackBuffer((T*)sbuf.ptr, sbuf.map, data, pdm, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPMDevCall::PackBuffer((T*)(packerptr[fid]), sbuf.map, data, pdm, block_dim, fstream);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid) && buffer_hdctype == HDCType::Host) {
            falmErrCheckMacro(falmMemcpyAsync(buffer[fid * 2].ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, (stream)? stream[fid] : 0));
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPM_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            CPM_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            }
        }
    }
}

template<typename T> void CPMComm<T>::IExchange6ColoredFace(T *data, INT color, INT thick, INT margin, int grp_tag, STREAM *stream) {
    Region &pdm = base->pdm_list[base->rank];
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferShapeOffset(thick, margin);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.allocColored(sizeof(T), sbuf.map, color, BufType::Out, buffer_hdctype, pdm);
            rbuf.allocColored(sizeof(T), rbuf.map, color, BufType::In , buffer_hdctype, pdm);
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * sbuf.count));
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CPMBuffer &sbuf = buffer[fid * 2];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPMDevCall::PackColoredBuffer((T*)sbuf.ptr, sbuf.map, color, data, pdm, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPMDevCall::PackColoredBuffer((T*)(packerptr[fid]), sbuf.map, color, data, pdm, block_dim, fstream);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid) && buffer_hdctype == HDCType::Host) {
            falmErrCheckMacro(falmMemcpyAsync(buffer[fid * 2].ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, (stream)? stream[fid] : 0));
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPM_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            CPM_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            }
        }
    }
}

template<typename T> void CPMComm<T>::PostExchange6Face(STREAM *stream) {
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * rbuf.count));
                falmErrCheckMacro(falmMemcpyAsync(packerptr[fid], rbuf.ptr, sizeof(T) * rbuf.count, MCpType::Hst2Dev, (stream)? stream[fid] : (STREAM)0));
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CPMBuffer &rbuf = buffer[fid * 2 + 1];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPMDevCall::UnpackBuffer((T*)rbuf.ptr, rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPMDevCall::UnpackBuffer((T*)(packerptr[fid]), rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.release();
            rbuf.release();
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            }
        }
    }
}

template<typename T> void CPMComm<T>::PostExchange6ColoredFace(STREAM *stream) {
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * rbuf.count));
                falmErrCheckMacro(falmMemcpyAsync(packerptr[fid], rbuf.ptr, sizeof(T) * rbuf.count, MCpType::Hst2Dev, (stream)? stream[fid] : (STREAM)0));
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CPMBuffer &rbuf = buffer[fid * 2 + 1];
            STREAM fstream = (stream)? stream[fid] : (STREAM)0;
            if (buffer_hdctype == HDCType::Device) {
                CPMDevCall::UnpackColoredBuffer((T*)rbuf.ptr, rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDCType::Host) {
                CPMDevCall::UnpackColoredBuffer((T*)(packerptr[fid]), rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            sbuf.release();
            rbuf.release();
            if (buffer_hdctype == HDCType::Host) {
                falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            }
        }
    }
}

}

#endif
