#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

make
make unit_build

#Qt testing requires an X-server to work. 
#Xvfb :42 -ac -screen 1920x1200  &
Xvfb :42 -ac &
sleep 5
export DISPLAY=:42
#icewm >/dev/null 2>&1 &
icewm  &
sleep 5

make check

kill `cat /tmp/.X42-lock`

if [[ $builddir == *"coverage"* ]]; then
  make coverage_info
  python $WORKSPACE/src/Externals/jenkins/lcov_cobertura.py coverage.info -o $WORKSPACE/build/coverage.xml
fi


