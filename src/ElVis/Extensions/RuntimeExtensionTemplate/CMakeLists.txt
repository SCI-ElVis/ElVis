# The MIT License
#
# Copyright (c) 2006 Scientific Computing and Imaging Institute,
# University of Utah (USA)
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# This file provides the tools to configure an ElVis Runtime Extension.
# Before using this file to create an extension, make the changes indicated
# by the TODO comments below.

cmake_minimum_required(VERSION 2.8)

SET(ProjectName "RuntimeExtensionGenerator")

SET(RUNTIME_EXTENSION_NAME "" CACHE STRING "Name of the extension")

IF( NOT RUNTIME_EXTENSION_NAME STREQUAL "" )
CONFIGURE_FILE(CMakeLists.txt.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/CMakeLists.txt @ONLY)
CONFIGURE_FILE(ExtensionCudaInterface.cu.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/ExtensionCudaInterface.cu @ONLY)
CONFIGURE_FILE(ExtensionOptiXInterface.cu.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/ExtensionOptiXInterface.cu @ONLY)
CONFIGURE_FILE(Model.cpp.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/Model.cpp @ONLY)
CONFIGURE_FILE(Model.h.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/Model.h @ONLY)
CONFIGURE_FILE(PluginInterface.cpp.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/PluginInterface.cpp @ONLY)
CONFIGURE_FILE(PluginInterface.h.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/PluginInterface.h @ONLY)
CONFIGURE_FILE(Declspec.h.in ${CMAKE_BINARY_DIR}/${RUNTIME_EXTENSION_NAME}/src/${RUNTIME_EXTENSION_NAME}/Declspec.h @ONLY)
ELSE()
    MESSAGE(SEND_ERROR "RUNTIME_EXTENSION_NAME must be specified.")
ENDIF()
