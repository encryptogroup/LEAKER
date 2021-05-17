cmake_minimum_required(VERSION 2.8.12)
project(pqtree_cpp)

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(pq-trees)

add_subdirectory(pybind11)
pybind11_add_module(pqtree_cpp wrappers.cpp pq-trees/pqtree.h pq-trees/pqtree.cc pq-trees/pqnode.h pq-trees/pqnode.cc )
