#!/bin/bash

WORKSPACE=$(git rev-parse --show-toplevel)

source $WORKSPACE/src/Externals/jenkins/jenkins_env.sh

cmake $WORKSPACE/src
