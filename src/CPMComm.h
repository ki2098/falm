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

static inline int CPM_Init(int *argc, char ***argv, CPM &cpm) {
    int err = MPI_Init(argc, argv);
    if (err != MPI_SUCCESS) return err;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &cpm.rank);
    if (err != MPI_SUCCESS) return err;
    err = MPI_Comm_size(MPI_COMM_WORLD, &cpm.size);
    if (cpm.use_cuda_aware_mpi) {
        cpm.bufman.hdc = HDC::Device;
    } else {
        cpm.bufman.hdc = HDC::Host;
    }
    return err;
}

static inline int CPM_Finalize(CPM &cpm) {
    cpm.bufman.release_all();
    return MPI_Finalize();
}

static inline int CPM_GetRank(MPI_Comm mpi_comm, int &mpi_rank) {
    return MPI_Comm_rank(mpi_comm, &mpi_rank);
}

static inline int CPM_GetSize(MPI_Comm mpi_comm, int &mpi_size) {
    return MPI_Comm_size(mpi_comm, &mpi_size);
}

static inline int CPM_ISend(CpmBuffer &buffer, MPI_Datatype mpi_dtype, int dst, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Isend(buffer.ptr, buffer.count, mpi_dtype, dst, tag, mpi_comm, mpi_req);
}

static inline int CPM_IRecv(CpmBuffer &buffer, MPI_Datatype mpi_dtype, int src, int tag, MPI_Comm mpi_comm, MPI_Request *mpi_req) {
    return MPI_Irecv(buffer.ptr, buffer.count, mpi_dtype, src, tag, mpi_comm, mpi_req);
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
    int            bufferId[12];
    Region         bufmap[12];
    MPI_Request         mpi_req[12];
    MPI_Status         mpi_stat[12];
    MPI_Datatype      mpi_dtype;
    T              *origin_ptr;
    Region        origin_domain;
    Flag         buffer_hdctype;
    void             *packerptr[6];
    

    CPMComm() : origin_ptr(nullptr) {}
    CPMComm(CPM *_base) : 
        base(_base),
        origin_ptr(nullptr)
    {
        mpi_dtype = getMPIDtype<T>();
        if (base->use_cuda_aware_mpi) {
            buffer_hdctype = HDC::Device;
        } else {
            buffer_hdctype = HDC::Host;
        }
    }

    void Wait6Face() {
        for (Int i = 0; i < 6; i ++) {
            if (base->neighbour[i] >= 0) {
                CPM_Waitall(2, &mpi_req[i * 2], &mpi_stat[i * 2]);
            }
        }
    }

    // void CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag);
    // void CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag);
    // void CPML2Dev_PostExchange6Face();
    // void CPML2Dev_PostExchange6ColoredFace();
    void IExchange6Face(T *data, Int thick, Int margin, int grp_tag, Stream *stream = nullptr);
    void IExchange6ColoredFace(T *data, Int color, Int thick, Int margin, int grp_tag, Stream *stream = nullptr);
    void PostExchange6Face(Stream *stream = nullptr);
    void PostExchange6ColoredFace(Stream *stream = nullptr);

protected:
    void makeBufferMaps(Int thick, Int margin) {
        Region &pdm = base->pdm_list[base->rank];
        for (Int fid = 0; fid < 6; fid ++) {
            // printf("%d\n", base->gc);
            if (base->neighbour[fid] >= 0) {
                Int gc = base->gc;
                Int gcx2 = 2 * gc;
                Int __s = fid*2, __r = fid*2+1;
                Int3 buffer_shape {
                    (fid == CPM::XPLUS || fid == CPM::XMINUS)? thick : pdm.shape[0] - gcx2,
                    (fid == CPM::YPLUS || fid == CPM::YMINUS)? thick : pdm.shape[1] - gcx2,
                    (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? thick : pdm.shape[2] - gcx2
                };
                Int3 sendbuffer_offset, recvbuffer_offset;
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
                bufmap[__s] = Region(buffer_shape, sendbuffer_offset);
                bufmap[__r] = Region(buffer_shape, recvbuffer_offset);
            }
        }
    }

};

template<typename T> void CPMComm<T>::IExchange6Face(T *data, Int thick, Int margin, int grp_tag, Stream *stream) {
    Region &pdm   = base->pdm_list[base->rank];
    CpmBufMan &bufman = base->bufman;
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferMaps(thick, margin);
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            Int sid = fid*2, rid = fid*2+1;
            // CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            // sbuf.alloc(sizeof(T), sbuf.map, BufType::Out, buffer_hdctype);
            // rbuf.alloc(sizeof(T), rbuf.map, BufType::In , buffer_hdctype);
            bufman.request(sizeof(T), bufmap[sid], &bufferId[sid]);
            bufman.request(sizeof(T), bufmap[rid], &bufferId[rid]);
            CpmBuffer &sbuf = bufman.get(bufferId[sid]);
            CpmBuffer &rbuf = bufman.get(bufferId[rid]);
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            if (buffer_hdctype == HDC::Host) {
                // falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * sbuf.count));
                CPMDevCall::PackBuffer((T*)sbuf.packer, sbuf.map, data, pdm, block_dim, fstream);
                falmErrCheckMacro(falmMemcpyAsync(sbuf.ptr, sbuf.packer, sizeof(T) * sbuf.count, MCP::Dev2Hst, fstream));
            } else if (buffer_hdctype == HDC::Device) {
                CPMDevCall::PackBuffer((T*)sbuf.ptr, sbuf.map, data, pdm, block_dim, fstream);
            } 
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            // CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            // MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            // CPM_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            // CPM_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
            // if (buffer_hdctype == HDC::Host) {
            //     falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            // }
            Int sid = fid*2, rid = fid*2+1;
            CPM_ISend(bufman.get(bufferId[sid]), mpi_dtype, base->neighbour[fid], base->neighbour[fid]+grp_tag*12, MPI_COMM_WORLD, &mpi_req[sid]);
            CPM_IRecv(bufman.get(bufferId[rid]), mpi_dtype, base->neighbour[fid], base->rank          +grp_tag*12, MPI_COMM_WORLD, &mpi_req[rid]);
        }
    }
}

template<typename T> void CPMComm<T>::IExchange6ColoredFace(T *data, Int color, Int thick, Int margin, int grp_tag, Stream *stream) {
    Region &pdm = base->pdm_list[base->rank];
    CpmBufMan &bufman = base->bufman;
    origin_ptr    = data;
    origin_domain = pdm;
    makeBufferMaps(thick, margin);
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            // CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            // sbuf.allocColored(sizeof(T), sbuf.map, color, BufType::Out, buffer_hdctype, pdm);
            // rbuf.allocColored(sizeof(T), rbuf.map, color, BufType::In , buffer_hdctype, pdm);
            Int sid = fid*2, rid = fid*2+1;
            bufman.request(sizeof(T), bufmap[sid], pdm, color, &bufferId[sid]);
            bufman.request(sizeof(T), bufmap[rid], pdm, color, &bufferId[rid]);
            CpmBuffer &sbuf = bufman.get(bufferId[sid]);
            CpmBuffer &rbuf = bufman.get(bufferId[rid]);
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            if (buffer_hdctype == HDC::Host) {
                // falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * sbuf.count));
                CPMDevCall::PackColoredBuffer((T*)sbuf.packer, sbuf.map, color, data, pdm, block_dim, fstream);
                falmErrCheckMacro(falmMemcpyAsync(sbuf.ptr, sbuf.packer, sizeof(T) * sbuf.count, MCP::Dev2Hst, fstream));
            } else if (buffer_hdctype == HDC::Device) {
                CPMDevCall::PackColoredBuffer((T*)sbuf.ptr, sbuf.map, color, data, pdm, block_dim, fstream);
            } 
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            // CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            // MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            // CPM_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            // CPM_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
            // if (buffer_hdctype == HDC::Host) {
            //     falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            // }
            Int sid = fid*2, rid = fid*2+1;
            CPM_ISend(bufman.get(bufferId[sid]), mpi_dtype, base->neighbour[fid], base->neighbour[fid]+grp_tag*12, MPI_COMM_WORLD, &mpi_req[sid]);
            CPM_IRecv(bufman.get(bufferId[rid]), mpi_dtype, base->neighbour[fid], base->rank          +grp_tag*12, MPI_COMM_WORLD, &mpi_req[rid]);
        }
    }
}

template<typename T> void CPMComm<T>::PostExchange6Face(Stream *stream) {
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CpmBuffer &rbuf = base->bufman.get(bufferId[fid*2+1]);
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            if (buffer_hdctype == HDC::Host) {
                // falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * rbuf.count));
                falmErrCheckMacro(falmMemcpyAsync(rbuf.packer, rbuf.ptr, sizeof(T) * rbuf.count, MCP::Hst2Dev, fstream));
                CPMDevCall::UnpackBuffer((T*)rbuf.packer, rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDC::Device) {
                CPMDevCall::UnpackBuffer((T*)rbuf.ptr, rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            // CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            // sbuf.release();
            // rbuf.release();
            // if (buffer_hdctype == HDC::Host) {
            //     falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            // }
            base->bufman.mark_release(bufferId[fid*2]);
            base->bufman.mark_release(bufferId[fid*2+1]);
        }
    }
}

template<typename T> void CPMComm<T>::PostExchange6ColoredFace(Stream *stream) {
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            dim3 block_dim(
                (fid == CPM::XPLUS || fid == CPM::XMINUS)? 1U : 8U,
                (fid == CPM::YPLUS || fid == CPM::YMINUS)? 1U : 8U,
                (fid == CPM::ZPLUS || fid == CPM::ZMINUS)? 1U : 8U
            );
            CpmBuffer &rbuf = base->bufman.get(bufferId[fid*2+1]);
            Stream fstream = (stream)? stream[fid] : (Stream)0;
            if (buffer_hdctype == HDC::Host) {
                // falmErrCheckMacro(falmMallocDevice((void**)&packerptr[fid], sizeof(T) * rbuf.count));
                falmErrCheckMacro(falmMemcpyAsync(rbuf.packer, rbuf.ptr, sizeof(T) * rbuf.count, MCP::Hst2Dev, fstream));
                CPMDevCall::UnpackColoredBuffer((T*)rbuf.packer, rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            } else if (buffer_hdctype == HDC::Device) {
                CPMDevCall::UnpackColoredBuffer((T*)rbuf.ptr, rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
            }
        }
    }
    if (!stream) {
        falmWaitStream(0);
    }
    for (Int fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (stream) {
                falmWaitStream(stream[fid]);
            }
            // CPMBuffer &sbuf = buffer[fid * 2], &rbuf = buffer[fid * 2 + 1];
            // sbuf.release();
            // rbuf.release();
            // if (buffer_hdctype == HDC::Host) {
            //     falmErrCheckMacro(falmFreeDevice(packerptr[fid]));
            // }
            base->bufman.mark_release(bufferId[fid*2]);
            base->bufman.mark_release(bufferId[fid*2+1]);
        }
    }
}

}

#endif
