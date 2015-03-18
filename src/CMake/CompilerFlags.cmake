#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2009 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software. 
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

#==================================================
# Default compiler flags, these can be modified under
# the advanced options using ccmake
#==================================================
IF( NOT PASSED_FIRST_CONFIGURE )

  #===============================
  # Set the build type to release by default, but debug if the binary directory contains the name debug
  SET( BUILD_TYPE_STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel Coverage." )

  IF( NOT CMAKE_BUILD_TYPE )
    #Get the name of the binary directory
    STRING( TOUPPER ${CMAKE_BINARY_DIR} BIN_DIR_NAME )
    STRING( FIND ${BIN_DIR_NAME} "/" LAST_DIR_IDX REVERSE )
    STRING( SUBSTRING ${BIN_DIR_NAME} LAST_DIR_IDX -1 BIN_DIR_NAME )

    IF( BIN_DIR_NAME MATCHES "DEBUG" )
      SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "COVERAGE_RELEASE" )
      SET(CMAKE_BUILD_TYPE "Coverage_Release" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "COVERAGE" )
      SET(CMAKE_BUILD_TYPE "Coverage" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSEIF( BIN_DIR_NAME MATCHES "RELEASE" )
      SET(CMAKE_BUILD_TYPE "Release" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
    ELSE()
      #SET(CMAKE_BUILD_TYPE "Release" CACHE STRING ${BUILD_TYPE_STRING} FORCE)
      SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING ${BUILD_TYPE_STRING} FORCE) #Default to debug for now
    ENDIF()
  ENDIF()
  

  #=============================

  #Compiler flags for the C++ compiler
  IF( CMAKE_COMPILER_IS_GNUCXX )

    SET( GNU_WARNING_FLAGS "-Werror -Wall -Wextra -Wno-unused-parameter -Wunused-result -Winit-self" )
    IF(GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
      SET( GNU_WARNING_FLAGS "${GNU_WARNING_FLAGS} -Wno-unused-local-typedefs" )
    ENDIF()

    SET( CMAKE_CXX_FLAGS "-std=c++0x ${GNU_WARNING_FLAGS} -fstrict-aliasing -Wstrict-aliasing" CACHE STRING "C++ Flags" FORCE)
    IF( NOT CYGWIN )
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" CACHE STRING "C++ Flags" FORCE)
    ENDIF()

    SET( CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ftrapv -fbounds-check" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops" CACHE STRING "C++ Release Flags" FORCE )

    SET( GNU_NO_INLINE_FLAGS "-fno-inline -fno-inline-functions -fno-inline-small-functions -fno-inline-functions-called-once -fno-default-inline -fno-implicit-inline-templates" )

    SET( CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage ${GNU_NO_INLINE_FLAGS}" CACHE STRING "C++ Coverage Flags" FORCE )
    SET( CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage" CACHE STRING "Executable Link Flags For Coverage Testing" FORCE )

    SET( CMAKE_CXX_FLAGS_COVERAGE_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --coverage ${GNU_NO_INLINE_FLAGS}" CACHE STRING "C++ Coverage Flags" FORCE )
    SET( CMAKE_EXE_LINKER_FLAGS_COVERAGE_RELEASE "--coverage" CACHE STRING "Executable Link Flags For Coverage Testing" FORCE )
  ELSEIF( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
    # -Wstrict-aliasing creates all kinds of crazy warnings for intel
    SET( CMAKE_CXX_FLAGS "-Werror -Wall -fPIC -fstrict-aliasing -ansi-alias-check" CACHE STRING "C++ Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_DEBUG "-g -O0" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_GUIDE "-parallel -guide-vec -guide-par" CACHE STRING "C++ Optimization Guide Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_ANALYSIS "-diag-enable sc3 -diag-enable vec" CACHE STRING "C++ Static Analysis Flags" FORCE )
  ELSEIF( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    #-Weverything -Wno-unreachable-code -Wno-newline-eof -Wno-c++98-compat -Wno-c++98-compat-pedantic 
    SET( CMAKE_CXX_FLAGS "-std=c++0x -Werror -Wall -fstrict-aliasing -Wstrict-aliasing -Wno-deprecated-register -Wno-deprecated-declarations" CACHE STRING "C++ Flags" FORCE)
    IF( APPLE )
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" CACHE STRING "C++ Flags" FORCE)
    ELSEIF( CYGWIN )
    ELSE()
      SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" CACHE STRING "C++ Flags" FORCE)
    ENDIF()
    SET( CMAKE_CXX_FLAGS_DEBUG "-g -O0 -flimit-debug-info -ftrapv" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_MEMCHECK "-g -O0 -flimit-debug-info -fsanitize=undefined -fbounds-checking -faddress-sanitizer -fno-omit-frame-pointer -fno-optimize-sibling-calls" CACHE STRING "C++ Debug Flags" FORCE )
    SET( CMAKE_CXX_FLAGS_ANALYSIS "--analyze" CACHE STRING "C++ Static Analysis Flags" FORCE )

    SET( CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage" CACHE STRING "C++ Release Flags" FORCE )
    SET( CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage" CACHE STRING "Executable Link Flags For Coverage Testing" FORCE )

  ENDIF()
  

  IF( MPI_CXX_FOUND )
    SET(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  ENDIF()
  
ENDIF()

MARK_AS_ADVANCED( FORCE
                  CMAKE_CXX_FLAGS_DEBUG
                  CMAKE_CXX_FLAGS_RELEASE
                  CMAKE_CXX_FLAGS_GUIDE
                  CMAKE_CXX_FLAGS_COVERAGE 
                  CMAKE_CXX_FLAGS_MEMCHECK
                  CMAKE_EXE_LINKER_FLAGS_COVERAGE 
                  CMAKE_CXX_FLAGS_COVERAGE_RELEASE
                  CMAKE_EXE_LINKER_FLAGS_COVERAGE_RELEASE 
                  CMAKE_CXX_FLAGS_VECTORIZE 
                  CMAKE_CXX_FLAGS_ANALYSIS
                )

IF( CMAKE_BUILD_TYPE )
  STRING( TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE )
  MARK_AS_ADVANCED( CLEAR CMAKE_CXX_FLAGS )
  MARK_AS_ADVANCED( CLEAR CMAKE_CXX_FLAGS_${BUILD_TYPE} )
ENDIF()

