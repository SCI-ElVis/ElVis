#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

make
make install

if [[ $buildnode == "cleopatra"* ]]; then
  /usr/local/VirtualGL/bin/vglrun make regcheck CTESTARGS="-T Test"
fi

if [[ $buildnode == "oci-ubuntu"* ]]; then
  /usr/bin/vglrun make regcheck CTESTARGS="-T Test"
fi

if [[ $buildnode == "Mac_node"* ]]; then
  make regcheck CTESTARGS="-T Test"
fi

git add $WORKSPACE/Testing/double/`uname`/
git commit -a -m "Updated regression testing images"
