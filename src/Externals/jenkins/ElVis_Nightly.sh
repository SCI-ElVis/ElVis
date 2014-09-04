#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/scripts/jenkins/jenkins_env.sh

cmake -DVALGRIND_EXTRA_FLAGS="--track-origins=yes;--xml=yes;--xml-file=$cmakedir/unit/valgrind.%p.memcheck.xml" \
      $WORKSPACE

#Qt testing requires an X-server to work. 
Xephyr :42 -ac -screen 1920x1200 >/dev/null 2>&1 &
sleep 5
DISPLAY=:42 icewm >/dev/null 2>&1 &

#This will check both dynamic and static memory
DISPLAY=:42 make MemAndStackCheck CTESTARGS="-T Test"
