#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

make
make unit_build
make check

if [[ $builddir == *"coverage"* ]]; then
  make coverage_info
  python $WORKSPACE/src/Externals/jenkins/lcov_cobertura.py coverage.info -o $WORKSPACE/build/coverage.xml
fi


