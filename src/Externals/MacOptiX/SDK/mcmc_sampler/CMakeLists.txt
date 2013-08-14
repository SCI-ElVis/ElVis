
#
#  Copyright (c) 2010 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

if(GLUT_FOUND AND OPENGL_FOUND)
  if(NOT CUDA_VERSION VERSION_LESS "3.0")
    include_directories(${GLUT_INCLUDE_DIR})
    add_definitions(-DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA)

    # See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
    OPTIX_add_sample_executable( mcmc_sampler
      mcmc_sampler.cpp

      # These files are common among multiple samples
      commonStructs.h
      helpers.h
      mcmc_draw_result.cu
      mcmc_sampler.cu
      mcmc_sampler.h
      random.h
      sunsky.cu
      sunsky.h

      # This is an extra option to compile to sm_20.  It is parsed out in the
      # OPTIX_add_sample_executable macro.
      OPTIONS -arch sm_20
    )

  else()
    first_time_message("Disabling mcmc_sampler which requires at least CUDA 3.0.  Currently using ${CUDA_VERSION}.")
  endif()

else()
  # GLUT or OpenGL not found
  first_time_message("Disabling mcmc_sampler, which requires glut and opengl.")
endif()
