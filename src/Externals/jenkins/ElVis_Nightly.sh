#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

cmake -DVALGRIND_EXTRA_FLAGS="--track-origins=yes;--xml=yes;--xml-file=$cmakedir/unit/valgrind.%p.memcheck.xml" \
      $WORKSPACE

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

if [[ $builddir == *"debug"* ]]; then
  #This will check both dynamic and static memory
  make MemAndStackCheck CTESTARGS="-T Test"
else
  #This will check dynamic memory
  make memheck CTESTARGS="-T Test"
fi

kill `cat /tmp/.X${DNUM}-lock`
