#ifndef FALM_CPMDEV_H
#define FALM_CPMDEV_H

#include "CPMB.h"

namespace Falm {

void dev_CPM_PackBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdom, dim3 &block_dim);

void dev_CPM_PackColoredBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdom, dim3 &block_dim);

void dev2hst_CPM_PackColoredBuffer(CPMBuffer<double> &buffer, double *src, Mapper &pdom, dim3 &block_dim);

void dev_CPM_UnpackBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdom, dim3 &block_dim);

void dev_CPM_UnpackColoredBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdom, dim3 &block_dim);

void hst2dev_CPM_UnpackColoredBuffer(CPMBuffer<double> &buffer, double *dst, Mapper &pdom, dim3 &block_dim);

}

#endif
