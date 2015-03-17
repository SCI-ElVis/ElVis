#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

source $WORKSPACE/src/Externals/jenkins/jenkins_env.sh

cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR -DELVIS_USE_DOUBLE_PRECISION=ON \
      $WORKSPACE/src
