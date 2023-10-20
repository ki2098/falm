#ifndef FALM_CPML2V2_H
#define FALM_CPML2V2_H

#include <mpi.h>
#include <typeinfo>
#include <vector>
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
    INT        gc;
    Region global;
    std::vector<Region> pdm_list;

    void initPartition(int mpi_rank, int mpi_size, INTx3 mpi_shape, INTx3 gShape, INT guideCell) {
        assert(mpi_size == PRODUCT3(mpi_shape));

        shape  = mpi_shape;
        rank   = mpi_rank;
        size   = mpi_size;
        initNeighbour();

        gc     = guideCell;
        global = Region(
            INTx3{gShape.x + gc*2, gShape.y + gc*2, gShape.z + gc*2},
            INTx3{0, 0, 0}
        );
        pdm_list = std::vector<Region>(size, Region());
        for (INT k = 0; k < shape.z; k ++) {
        for (INT j = 0; j < shape.y; j ++) {
        for (INT i = 0; i < shape.x; i ++) {
            INT ox = 0, oy = 0, oz = 0;
            for (INT __x = 0; __x < i; __x ++) {
                ox += dim_division(gShape.x, shape.x, __x);
            }
            for (INT __y = 0; __y < j; __y ++) {
                oy += dim_division(gShape.y, shape.y, __y);
            }
            for (INT __z = 0; __z < k; __z ++) {
                oz += dim_division(gShape.z, shape.z, __z);
            }
            pdm_list[IDX(i, j, k, shape)] = Region(
                INTx3{
                    dim_division(gShape.x, shape.x, i) + gc*2,
                    dim_division(gShape.y, shape.y, j) + gc*2,
                    dim_division(gShape.z, shape.z, k) + gc*2
                },
                INTx3{ox, oy, oz}
            );
        }}}

    }

    void set6Region(INTx3 &inner_shape, INTx3 &inner_offset, INTx3 *boundary_shape, INTx3 *boundary_offset, INT thick, const Region &map) {
        inner_shape = map.shape;
        inner_offset = map.offset;
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

    // void setDefaultRegions(INTx3 &inner_shape, INTx3 &inner_offset, INTx3 *boundary_shape, INTx3 *boundary_offset, INT thick, Mapper &pdm) {
    //     Mapper __map(pdm, gc);
    //     setNonDefaultRegions(inner_shape, inner_offset, boundary_shape, boundary_offset, thick, __map);
    // }

    bool validNeighbour(INT fid) {
        return (neighbour[fid] >= 0);
    }

protected:
    INT dim_division(INT dim_size, INT n_division, INT id) {
        INT p_dim_size = dim_size / n_division;
        if (id < dim_size % n_division) {
            p_dim_size ++;
        }
        return p_dim_size;
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

};



template<typename T>
class CPMOp {
public:
    CPMBase               *base;
    CPMBuffer            buffer[12];
    MPI_Request         mpi_req[12];
    MPI_Status         mpi_stat[12];
    MPI_Datatype      mpi_dtype;
    T              *origin_ptr;
    Region        origin_domain;
    FLAG         buffer_hdctype;
    void             *packerptr[6];
    

    CPMOp() : origin_ptr(nullptr) {}
    CPMOp(CPMBase *_base) : 
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

    void CPML2_Wait6Face() {
        for (INT i = 0; i < 6; i ++) {
            if (base->neighbour[i] >= 0) {
                CPML2_Waitall(2, &mpi_req[i * 2], &mpi_stat[i * 2]);
            }
        }
    }

    // void CPML2Dev_IExchange6Face(T *data, Mapper &pdm, INT thick, int grp_tag);
    // void CPML2Dev_IExchange6ColoredFace(T *data, Mapper &pdm, INT color, INT thick, int grp_tag);
    // void CPML2Dev_PostExchange6Face();
    // void CPML2Dev_PostExchange6ColoredFace();
    void CPML2Dev_IExchange6Face(T *data, INT thick, INT margin, int grp_tag, STREAM *stream = nullptr);
    void CPML2Dev_IExchange6ColoredFace(T *data, INT color, INT thick, INT margin, int grp_tag, STREAM *stream = nullptr);
    void CPML2Dev_PostExchange6Face(STREAM *stream = nullptr);
    void CPML2Dev_PostExchange6ColoredFace(STREAM *stream = nullptr);

protected:
    void makeBufferShapeOffset(INT thick, INT margin) {
        Region &pdm = base->pdm_list[base->rank];
        for (INT fid = 0; fid < 6; fid ++) {
            // printf("%d\n", base->gc);
            if (base->neighbour[fid] >= 0) {
                INT gc = base->gc;
                INT gcx2 = 2 * gc;
                INT __s = fid*2, __r = fid*2+1;
                INTx3 buffer_shape {
                    (fid / 2 == 0)? thick : pdm.shape.x - gcx2,
                    (fid / 2 == 1)? thick : pdm.shape.y - gcx2,
                    (fid / 2 == 2)? thick : pdm.shape.z - gcx2
                };
                INTx3 sendbuffer_offset, recvbuffer_offset;
                if (fid == 0) {
                    sendbuffer_offset = {pdm.shape.x - gc - thick - margin, gc, gc};
                    recvbuffer_offset = {pdm.shape.x - gc         + margin, gc, gc};
                } else if (fid == 1) {
                    sendbuffer_offset = {              gc         + margin, gc, gc};
                    recvbuffer_offset = {              gc - thick - margin, gc, gc};
                } else if (fid == 2) {
                    sendbuffer_offset = {gc, pdm.shape.y - gc - thick - margin, gc};
                    recvbuffer_offset = {gc, pdm.shape.y - gc         + margin, gc};
                } else if (fid == 3) {
                    sendbuffer_offset = {gc,               gc         + margin, gc};
                    recvbuffer_offset = {gc,               gc - thick - margin, gc};
                } else if (fid == 4) {
                    sendbuffer_offset = {gc, gc, pdm.shape.z - gc - thick - margin};
                    recvbuffer_offset = {gc, gc, pdm.shape.z - gc         + margin};
                } else if (fid == 5) {
                    sendbuffer_offset = {gc, gc,               gc         + margin};
                    recvbuffer_offset = {gc, gc,               gc - thick - margin};
                }
                buffer[__s].map = Region(buffer_shape, sendbuffer_offset);
                buffer[__r].map = Region(buffer_shape, recvbuffer_offset);
            }
        }
    }

};

template<typename T> void CPMOp<T>::CPML2Dev_IExchange6Face(T *data, INT thick, INT margin, int grp_tag, STREAM *stream) {
    assert(origin_ptr == nullptr);
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
                packerptr[fid] = falmMallocDevice(sizeof(T) * sbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
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
                CPML0Dev_PackBuffer((T*)(packerptr[fid]), sbuf.map, data, pdm, block_dim, fstream);
                falmMemcpyAsync(sbuf.ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, fstream);
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
            if (buffer_hdctype == HDCType::Host) {
                falmFreeDevice(packerptr[fid]);
            }
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPML2_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            CPML2_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2Dev_IExchange6ColoredFace(T *data, INT color, INT thick, INT margin, int grp_tag, STREAM *stream) {
    assert(origin_ptr == nullptr);
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
                packerptr[fid] = falmMallocDevice(sizeof(T) * sbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
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
                CPML0Dev_PackColoredBuffer((T*)(packerptr[fid]), sbuf.map, color, data, pdm, block_dim, fstream);
                falmMemcpyAsync(sbuf.ptr, packerptr[fid], sizeof(T) * sbuf.count, MCpType::Dev2Hst, fstream);
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
            if (buffer_hdctype == HDCType::Host) {
                falmFreeDevice(packerptr[fid]);
            }
            fflush(stdout);
            CPMBuffer   &sbuf =  buffer[fid * 2], &rbuf =  buffer[fid * 2 + 1];
            MPI_Request &sreq = mpi_req[fid * 2], &rreq = mpi_req[fid * 2 + 1];
            CPML2_ISend(sbuf, mpi_dtype, base->neighbour[fid], base->neighbour[fid] + grp_tag * 12, MPI_COMM_WORLD, &sreq);
            CPML2_IRecv(rbuf, mpi_dtype, base->neighbour[fid], base->rank           + grp_tag * 12, MPI_COMM_WORLD, &rreq);
        }
    }
}

template<typename T> void CPMOp<T>::CPML2Dev_PostExchange6Face(STREAM *stream) {
    assert(origin_ptr != nullptr);
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                packerptr[fid] = falmMallocDevice(sizeof(T) * rbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
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
                CPML0Dev_UnpackBuffer((T*)(packerptr[fid]), rbuf.map, origin_ptr, origin_domain, block_dim, fstream);
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
        if (base->validNeighbour(fid)) {
            if (buffer_hdctype == HDCType::Host) {
                CPMBuffer &rbuf = buffer[fid * 2 + 1];
                packerptr[fid] = falmMallocDevice(sizeof(T) * rbuf.count);
            }
        }
    }
    for (INT fid = 0; fid < 6; fid ++) {
        if (base->validNeighbour(fid)) {
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
                CPML0Dev_UnpackColoredBuffer((T*)(packerptr[fid]), rbuf.map, rbuf.color, origin_ptr, origin_domain, block_dim, fstream);
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

}

#endif
