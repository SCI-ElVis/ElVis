#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

source $WORKSPACE/src/Externals/jenkins/jenkins_env.sh

if [[ `hostname` == "cleopatra"* ]]; then

  PREFIX=

elif [[ `hostname` == "oci-ubuntu"* ]]; then

  PREFIX="-DCMAKE_PREFIX_PATH=/home/jenkins/nektar/build/dist/lib64/nektar++-4.0.1/cmake"

elif [[ `hostname` == "dhcp"* ]]; then

  PREFIX="-DCMAKE_PREFIX_PATH=/Users/jenkins/nektar/build/dist/lib/nektar++-4.0.1/cmake"

else
  echo "Please configure `hostname` in src/Externals/jenkins/cmake_jenkins.sh"
  exit 1
fi


cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DELVIS_USE_DOUBLE_PRECISION=ON \
      -DELVIS_ENABLE_ProjectX_EXTENSION=ON \
      -DELVIS_ENABLE_NEKTAR++_EXTENSION=ON \
      $PREFIX  \
      $WORKSPACE/src
