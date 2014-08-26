#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

cmakedir=$WORKSPACE/build/$builddir

rm -rf $WORKSPACE/build
mkdir -p $cmakedir
cd $cmakedir

source $WORKSPACE/scripts/jenkins/jenkins_env.sh

cmake -DVALGRIND_EXTRA_FLAGS="--track-origins=yes;--xml=yes;--xml-file=$cmakedir/unit/valgrind.%p.memcheck.xml" \
      $WORKSPACE

#This will check both dynamic and static memory
make MemAndStackCheck CTESTARGS="-T Test"
