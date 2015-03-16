#!/bin/bash

shopt -s nocasematch

dir=`pwd`
builddir=`basename $dir`

if [[ `hostname` == "cleopatra"* || `hostname` == "oci-ubuntu"* ]]; then

  export OptiX_INSTALL_DIR=/usr/local/NVIDIA-OptiX-SDK-3.6.0-linux64
  export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-6.0/

elif [[ `hostname` == "dhcp227"* ]]; then

  export OptiX_INSTALL_DIR=/Developer/OptiX
  export CUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-6.0

else
  echo "Please configure `hostname` in src/Externals/jenkins/jenkins_env.sh"
  exit 1
fi

env > env.log 2>&1
