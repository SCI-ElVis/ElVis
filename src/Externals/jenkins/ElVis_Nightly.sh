#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

cmake -DVALGRIND_EXTRA_FLAGS="--track-origins=yes;--xml=yes;--xml-file=$cmakedir/valgrind.%p.memcheck.xml" \
      $WORKSPACE/src

make
make install

if [[ $builddir == *"debug"* ]]; then
  #This will check both dynamic and static memory
  make MemAndStackCheck CTESTARGS="-T Test"
else
  #This will check dynamic memory
  make memcheck CTESTARGS="-T Test"
fi

