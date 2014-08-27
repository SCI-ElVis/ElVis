#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

#Qt testing requires an X-server to work. 
Xephyr :42 -ac -screen 1920x1200 >/dev/null 2>&1 &
sleep 5
DISPLAY=:42 icewm >/dev/null 2>&1 &

make
make unit_build
DISPLAY=:42 make check

if [[ $builddir == *"coverage"* ]]; then
  make coverage_info
  python $WORKSPACE/src/Externals/jenkins/lcov_cobertura.py coverage.info -o $WORKSPACE/build/coverage.xml
fi


