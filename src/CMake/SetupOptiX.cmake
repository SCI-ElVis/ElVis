# Creates various macros and performs other setup required for OptiX

FIND_PACKAGE(OptiX REQUIRED)

IF( OptiX_FOUND )
    # Setup OptiX related stuff
    SET(CMAKE_MODULE_PATH ${OptiX_INSTALL_DIR}/SDK/CMake ${CMAKE_MODULE_PATH})

    # Include all CMake Macros.
    include(Macros)
    # Determine information about the compiler
    include (CompilerInfo)
    # Check for specific machine/compiler options.
    include (ConfigCompilerFlags)

    # Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
    # is a custom extension to the FindCUDA code distributed by CMake.
    OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

    # Find at least a 3.0 version of CUDA.
    find_package(CUDA 3.0 REQUIRED)

    #INCLUDE_DIRECTORIES(${OptiX_INCLUDE} ${CUDA_TOOLKIT_INCLUDE})

    # Cuda 5.0 does not have the SDK, but instead has a samples directory.
    #IF( NOT( CUDA_VERSION VERSION_LESS "5.0") )

    #  FIND_PATH(CUDA_SAMPLE_DIR 
    #      common/inc/helper_cuda.h
    #      PATHS
    #      "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}"
    #      /Developer/NVIDIA/CUDA-${CUDA_VERSION}/samples
    #      )

    #  IF( NOT CUDA_SAMPLE_DIR )
    #    MESSAGE(FATAL_ERROR "Error - CUDA_SAMPLE_DIR must be set.")
    #  ENDIF()
      # The default find cuda will find older versions of the SDK if multiple 
      # cuda versions are installed.
    #  SET(CUDA_SDK_ROOT_DIR "" CACHE PATH "" FORCE)
    #ENDIF()


    IF( CUDA_VERSION VERSION_LESS "5.0" )
        #INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
        SET(ELVIS_CUDA_INCLUDE ${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/C/common/inc)
    ELSE()
        #INCLUDE_DIRECTORIES(${CUDA_SAMPLE_DIR}/common/inc)
        SET(ELVIS_CUDA_INCLUDE ${CUDA_TOOLKIT_INCLUDE} ${CUDA_SAMPLE_DIR}/common/inc)
    ENDIF()

    get_filename_component(path_to_optix "${optix_LIBRARY}" PATH)
    #set_property( DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY 
     #             LINK_DIRECTORIES ${path_to_optix} )
    set( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${path_to_optix} )

# Add some useful default arguments to the nvcc flags.  This is an example of how we use
# PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE and following
# block of code will not be executed leaving you free to edit the values as much as you
# wish from the GUI or from ccmake.
if(NOT OPTIX_PASSED_FIRST_CONFIGURE)
  set(flag "--use_fast_math")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()

  set(flag "--keep")
  list(FIND CUDA_NVCC_FLAGS ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS ${flag})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
  endif()
  
  if( APPLE )
    # Undef'ing __BLOCKS__ for OSX builds.  This is due to a name clash between OSX 10.6
    # C headers and CUDA headers
    set(flag "-U__BLOCKS__")
    list(FIND CUDA_NVCC_FLAGS ${flag} index)
    if(index EQUAL -1)
      list(APPEND CUDA_NVCC_FLAGS ${flag})
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    endif()
  endif()
  
  # This passes a preprocessor definition to cl.exe when processing CUDA code.
  if(USING_WINDOWS_CL)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
  endif()

endif(NOT OPTIX_PASSED_FIRST_CONFIGURE)

set(OPTIX_PASSED_FIRST_CONFIGURE ON CACHE BOOL INTERNAL)
mark_as_advanced(OPTIX_PASSED_FIRST_CONFIGURE)

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
function(ADD_OPTIX_EXECUTABLE target_name ptx_dir)

  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Create the rules to build the PTX from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options}
    OPTIONS ${options} )

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_executable(${target_name}
    ${source_files}
    #${generated_files}
    ${cmake_options}
    )

  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    #sutil
    optix
    ${optix_LIBRARY}
    ${optix_rpath}
    )

endfunction()

#########################################################
# OPTIX_add_sample_executable
#
# Convience function for adding samples to the code.  You can copy the contents of this
# funtion into your individual project if you wish to customize the behavior.  Note that
# in CMake, functions have their own scope, whereas macros use the scope of the caller.
#
# ptx_dir specifies where the compiled .cu files are placed.
function(ADD_OPTIX_LIBRARY target_name ptx_dir)

  set(CUDA_GENERATED_OUTPUT_DIR ${ptx_dir})

  if (WIN32)
    string(REPLACE "\\" "\\\\" CUDA_GENERATED_OUTPUT_DIR ${CUDA_GENERATED_OUTPUT_DIR})
    set(PATH_SEPARATOR "\\\\")
  else (WIN32)
    set(PATH_SEPARATOR "/")
  endif (WIN32)
  
  # These calls will group PTX and CUDA files into their own directories in the Visual
  # Studio projects.
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

  # Separate the sources from the CMake and CUDA options fed to the macro.  This code
  # comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
  # code here, so that we can use our own name for the target.  target_name is used in the
  # creation of the output file names, and we want this to be unique for each target in
  # the SDK.
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Create the rules to build the PTX from the CUDA files.
  CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options}
    OPTIONS ${options} )

  # Here is where we create the rule to make the executable.  We define a target name and
  # list all the source files used to create the target.  In addition we also pass along
  # the cmake_options parsed out of the arguments.
  add_library(${target_name} SHARED
    ${source_files}
    #${generated_files}
    ${cmake_options}
    )
  
  # Most of the samples link against the sutil library and the optix library.  Here is the
  # rule that specifies this linkage.
  target_link_libraries( ${target_name}
    #sutil
    optix
    ${optix_rpath}
    )
    
endfunction()

ENDIF()
