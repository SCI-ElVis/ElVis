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

FIND_PACKAGE(OptiX REQUIRED)

macro(OPTIX_add_sample_shared_library target_name)
  ADD_ELVIS_LIBRARY(${target_name} SHARED
    ${ARGN}
    )

  set(cuda_file_list $ENV{cuda_file_list})
  foreach(file ${ARGN})
    #message("file = ${file}")
    if(file MATCHES ".*cu$")
      list(APPEND cuda_file_list ${file})
    endif()
  endforeach()
  # Remove duplicates to keep the list small
  list(REMOVE_DUPLICATES cuda_file_list)
  # Don't forget the quotes around ${cuda_file_list}, otherwise you will only
  # get the first item in the list set.
  set(ENV{cuda_file_list} "${cuda_file_list}")

  target_link_libraries( ${target_name}
    ${SUTIL_LIB}
    optix
    )

  #add_dependencies(${target_name} sample-ptx)

  #add_perforce_to_target( ${target_name} )

endmacro()
set(ENV{cuda_file_list})