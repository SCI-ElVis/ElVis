#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

make
make unit_build

if [[ $builddir == "debug" ]]; then
  DNUM=42
elif [[ $builddir == "release" ]]; then
  DNUM=52
elif [[ $builddir == "coverage" ]]; then
  DNUM=62
fi

#Qt testing requires an X-server to work. 
#Xvfb :42 -ac -screen 1920x1200  &
export DISPLAY=:$DNUM
Xvfb :$DNUM -ac > /dev/null 2>&1 &
sleep 5
icewm >/dev/null 2>&1 &
sleep 5

make check

kill `cat /tmp/.X${DNUM}-lock`

if [[ $builddir == *"coverage"* ]]; then
  make coverage_info
  python $WORKSPACE/src/Externals/jenkins/lcov_cobertura.py coverage.info -o $WORKSPACE/build/coverage.xml
fi


