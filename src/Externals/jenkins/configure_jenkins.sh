#!/bin/bash

flags=(debug release)

if [[ `uname -a` == "Darwin"* ]]; then
  compilers=(clang)
else
  compilers=(gnu)
fi

WORKSPACE=$(git rev-parse --show-toplevel)

if [ -d $WORKSPACE/build ]; then
  echo 
  echo "Please remove the directory '$WORKSPACE/build' before running this script"
  echo
  exit 1
fi

mkdir $WORKSPACE/build


echo "flags = ${flags[@]}" >> $WORKSPACE/build/builds.make
echo "compilers = ${compilers[@]}" >> $WORKSPACE/build/builds.make

#Create the coverage directory
mkdir $WORKSPACE/build/coverage
cd $WORKSPACE/build/coverage
$WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh

#Create all the other ones
for flag in ${flags[@]}
do
  for compiler in ${compilers[@]}
  do
    mkdir $WORKSPACE/build/${flag}_${compiler}
    cd $WORKSPACE/build/${flag}_${compiler}
    $WORKSPACE/src/Externals/jenkins/cmake_jenkins.sh
  done
done

cp $WORKSPACE/src/Externals/jenkins/Makefile.jenkins $WORKSPACE/build/Makefile

