/*
For License information see the LICENSE file.

Authors: Abdelkarim Kati
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/stl_bind.h> 
#include <cstdio>
#include <iostream>
#include <assert.h>
#include <set>
#include "pqtree.h"
#include "pqnode.h"
PYBIND11_MAKE_OPAQUE(std::vector<PQNode*>);
PYBIND11_MAKE_OPAQUE(std::map<int, PQNode*>);


namespace py = pybind11;

PYBIND11_MODULE(pqtree_cpp, m)
{
    m.doc() = "C++ PQTree implementation wrappers"; 

    py::class_<PQNode, std::unique_ptr<PQNode, py::nodelete>>(m, "PQNode")
        .def(py::init<>())
        .def("Type", &PQNode::Type)
        .def("LeafValue", &PQNode::LeafValue)
        .def("FindLeaves", &PQNode::FindLeaves)
        .def("Children", &PQNode::Children); 
    py::enum_<PQNode::PQNode_types>(m, "PQNode_types")
        .value("leaf", PQNode::PQNode_types::leaf)
        .value("pnode", PQNode::PQNode_types::pnode)
        .value("qnode", PQNode::PQNode_types::qnode)
        .export_values();

    py::bind_vector<std::vector<PQNode*>>(m, "PQNodeArray");
    py::bind_map<std::map<int, PQNode*>>(m, "PQNodeDict");


    py::class_<PQTree>(m, "PQTree")
        .def(py::init<set<int>>())              
        .def("SafeReduce", &PQTree::SafeReduce)
        .def("SafeReduceAll", &PQTree::SafeReduceAll)
        .def("Reduce", &PQTree::Reduce)
        .def("ReduceAll", &PQTree::ReduceAll)


        .def("Root", &PQTree::Root)
        .def("Print", &PQTree::Print)
        .def("CleanPseudo", &PQTree::CleanPseudo)
        .def("Frontier", &PQTree::Frontier)
        .def("ReducedFrontier", &PQTree::ReducedFrontier)
        .def("GetReductions", &PQTree::GetReductions)
        .def("GetContained", &PQTree::GetContained);
}
