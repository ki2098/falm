#ifndef _STRUCTURED_MESH_H_
#define _STRUCTURED_MESH_H_ 1

#include "StructuredField.cuh"
#include "Dom.cuh"

namespace FALM {

struct Mesh {
    Field<double> x;
    Field<double> h;
    Field<double> v;

    Mesh(Dom &dom);
    void sync_h2d();
    void sync_d2h();
    void release(int loc);
};

Mesh::Mesh(Dom &dom) : x(dom._h._size, 3, FALMLoc::HOST, 1), h(dom._h._size, 3, FALMLoc::HOST, 2), v(dom._h._size, 3, FALMLoc::HOST, 3) {}

void Mesh::sync_d2h() {
    x.sync_d2h();
    h.sync_d2h();
    v.sync_d2h();
}

void Mesh::sync_h2d() {
    x.sync_h2d();
    h.sync_h2d();
    v.sync_h2d();
}

void Mesh::release(int loc) {
    x.release(loc);
    h.release(loc);
    v.release(loc);
}

}

#endif