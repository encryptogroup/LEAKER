#!/bin/sh

cd `dirname $0`

OLDIFS=$IFS
IFS=','

for i in https://github.com/pybind/pybind11,pybind11 https://github.com/Gregable/pq-trees,pq-trees; do
    set -- $i
    if [ ! -d "$2" ] ; then
        git clone $1 $2
    fi
done
IFS=$OLDIFS


grep -v "private:" pq-trees/pqnode.h > tmp; mv tmp pq-trees/pqnode.h
grep -v "private:" pq-trees/pqtree.h > tmp; mv tmp pq-trees/pqtree.h
sed  -i '/class PQTree {/a public:' pq-trees/pqtree.h
mkdir -p build
cd build
cmake .. -Wno-dev
make