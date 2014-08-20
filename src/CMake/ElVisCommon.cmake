

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