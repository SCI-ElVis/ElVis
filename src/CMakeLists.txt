IF( APPLE )
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
ELSE()
  CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8 FATAL_ERROR)
ENDIF()

# As of CMake 2.6 policies were introduced in order to provide a mechanism for
# adding backwards compatibility one feature at a time.  We will just specify
# that all policies will use version 2.6 semantics.
cmake_policy(VERSION 2.6)

#Set the apple compiler to default to clang, that's all apple supports anyways...
IF ( APPLE )
    SET(CMAKE_CXX_COMPILER clang++)
    SET(CMAKE_C_COMPILER clang)
    SET(CUDA_HOST_COMPILER clang)
    SET(Boost_NO_BOOST_CMAKE ON CACHE INTERNAL "Defaults to ON so ElVis' included Boost is used rather than installed version.")
    SET(Boost_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/Externals/boost CACHE PATH "Path to ElVis boost includes")
    SET(Boost_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/Externals/boost/libs CACHE PATH "Path to ElVis boost libs")
ENDIF ()

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

#IF( NOT(CMAKE_INSTALL_PREFIX_SET) )
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "Install Path" FORCE)
#ENDIF()
#SET(CMAKE_INSTALL_PREFIX_SET ON CACHE INTERNAL "")

#Add the bin directory to the rpath
SET( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/bin )
            
PROJECT(ElVis)

ENABLE_TESTING()
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/CMake)

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
#IF(NOT CMAKE_BUILD_TYPE AND NOT PASSED_FIRST_CONFIGURE)
#  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
#      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
#      FORCE)
#ENDIF()

SET(ELVIS_VERSION 0.1.1)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/bin ELVIS_BIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/cubin ELVIS_CUBIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/plugins ELVIS_PLUGIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/lib ELVIS_LIB_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/include ELVIS_INCLUDE_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/ColorMaps ELVIS_COLOR_MAP_DIR)

FIND_PACKAGE(Qt4 REQUIRED)

IF( QT4_FOUND )
    INCLUDE(${QT_USE_FILE})
ENDIF()

SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMake )

INCLUDE(CMake/SetupOptiX.cmake)

#Need to set compiler flags after SetupOptix
INCLUDE(CMake/CompilerFlags.cmake)

#Need to create unit tests after compiler flags have been set
INCLUDE(CMake/UnitTest.cmake)

ADD_SUBDIRECTORY(Externals)

IF( NOT PASSED_FIRST_CONFIGURE )
    IF( CUDA_VERSION VERSION_LESS "5.0" )
      LIST(APPEND CUDA_NVCC_FLAGS_DEBUG "-G0")
    ENDIF()
    SET(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    
    LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    
    SET(CUDA_BUILD_CUBIN ON CACHE BOOL "" FORCE)
ENDIF() 

OPTION(ELVIS_USE_DOUBLE_PRECISION "Use double precision value in GPU code.")
OPTION(ELVIS_ENABLE_PRINTF "Enable the use of ELVIS_PRINTF in device code.")

IF( ELVIS_ENABLE_PRINTF )
    ADD_DEFINITIONS(-DELVIS_ENABLE_PRINTF)
ENDIF()

SET(GLUT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Externals/glut/include)

# {CMAKE_BINARY_DIR} be included to handle generated files.  These files won't exist 
# in CVS and should not be put in the src directory.
INCLUDE_DIRECTORIES( 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${Boost_INCLUDE_DIR}
    ${GLEW_INCLUDE}
    ${GLUT_INCLUDE_DIR}
  # Cuda includes must appear after other GL includes to prevent accidentally
  # including the versions that ship with cuda.
  ${OptiX_INCLUDE}
  ${ELVIS_CUDA_INCLUDE}
)


IF(ELVIS_USE_DOUBLE_PRECISION)
    SET(ELVIS_FLOAT "double")
ELSE()
    SET(ELVIS_FLOAT "float")
ENDIF()


SET(CUDA_GENERATED_OUTPUT_DIR ${ELVIS_CUBIN_DIR} CACHE PATH "" FORCE)

IF( MSVC )	
    # Removes the warnings about unsafe methods such as strcpy, std::copy, memcmp, etc.
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_DEPRECATE)

    # Adds support for M_PI
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)	
ENDIF( MSVC )

MACRO(SET_ELVIS_TARGET_PROPERTIES name)
    SET_TARGET_PROPERTIES(${name} PROPERTIES DEBUG_POSTFIX -g)
    SET_TARGET_PROPERTIES(${name} PROPERTIES MINSIZEREL_POSTFIX -ms)
    SET_TARGET_PROPERTIES(${name} PROPERTIES RELWITHDEBINFO_POSTFIX -rg)

    #SET(ADDITIONAL_COMPILE_FLAGS "-Werror -Wall")
    #get_target_property(ADDITIONAL_COMPILE_FLAGS ${name} COMPILE_FLAGS)
    #IF( ${ADDITIONAL_COMPILE_FLAGS} STREQUAL "ADDITIONAL_COMPILE_FLAGS-NOTFOUND" )
    #    SET(ADDITIONAL_COMPILE_FLAGS "")
    #ENDIF()
    
    IF( CMAKE_COMPILER_IS_GNUCXX )
    #    set(ADDITIONAL_COMPILE_FLAGS "${ADDITIONAL_COMPILE_FLAGS} -std=c++0x")
	IF( Boost_FOUND )
	  IF( ${Boost_MAJOR_VERSION} LESS 2 AND
	      ${Boost_MINOR_VERSION} LESS 51)
	    ADD_DEFINITIONS(-DBOOST_NO_SCOPED_ENUMS)
	  ELSE()
	    ADD_DEFINITIONS(-DBOOST_NO_CXX11_SCOPED_ENUMS)
	  ENDIF() 
	ENDIF()
    #    # Some versions of gcc can't handle -g3, but it is set by a cuda
    #    # CMakeLists
    #    SET(STR1 " ")
    #    SET(STR2 " ")
    #    STRING(REPLACE "-g3" "" STR1 ${CMAKE_CXX_FLAGS_DEBUG} )
    #    STRING(REPLACE "-g" "" STR2 ${STR1} )
    #    SET(CMAKE_CXX_FLAGS_DEBUG "${STR2} -g" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  
    #    STRING(REPLACE "-g3" "" STR3 ${CMAKE_CXX_FLAGS_RELEASE} )
    #    STRING(REPLACE "-g" "" STR4 ${STR3} )
    #    SET(CMAKE_CXX_FLAGS_RELEASE "${STR4}" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    ENDIF()

    #IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        # The static libraries must be compiled with position independent code
        # on 64 bit Linux.
    #    SET(ADDITIONAL_COMPILE_FLAGS "${ADDITIONAL_COMPILE_FLAGS} -fPIC")
    #ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
    
    #IF( ADDITIONAL_COMPILE_FLAGS )
    #    SET_TARGET_PROPERTIES(${name} PROPERTIES COMPILE_FLAGS ${ADDITIONAL_COMPILE_FLAGS})
    #ENDIF()

    IF( MSVC )
        # Disable the warnings about duplicate copy/assignment methods
        #   (4521, 4522)
        # Disable the warning that arrays are default intialized (4351)
        # Disable "forcing value to bool 'true' or 'false' (performance
        #   warning)" warning (4800)

        # /Za is necessary to prevent temporaries being bound to reference
        #   parameters.
        SET_TARGET_PROPERTIES(${name} PROPERTIES COMPILE_FLAGS
                                "/wd4521 /wd4522 /wd4351 /wd4018 /wd4800 /wd4250")

        # Enable source level parallel builds.
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    ENDIF(  )
ENDMACRO(SET_ELVIS_TARGET_PROPERTIES name)

MACRO(ADD_ELVIS_EXECUTABLE name)
    ADD_EXECUTABLE(${name} ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )
ENDMACRO(ADD_ELVIS_EXECUTABLE name)

MACRO(ADD_CUDA_ELVIS_EXECUTABLE name)
    CUDA_ADD_EXECUTABLE(${name} ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )

ENDMACRO(ADD_CUDA_ELVIS_EXECUTABLE name)

MACRO(ADD_ELVIS_STATIC_LIBRARY name)
    ADD_LIBRARY(${name} STATIC ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR} 
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR} NAMELINK_SKIP
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )
ENDMACRO(ADD_ELVIS_STATIC_LIBRARY name)

MACRO(ADD_ELVIS_PLUGIN name)
    ADD_LIBRARY(${name} SHARED ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR} 
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR} NAMELINK_SKIP
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )
ENDMACRO()

MACRO(ADD_ELVIS_LIBRARY name type)
    ADD_LIBRARY(${name} ${type} ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR} 
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR} NAMELINK_SKIP
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR}  )
ENDMACRO(ADD_ELVIS_LIBRARY name)


MACRO(ADD_ELVIS_CUDA_LIBRARY name type)
    CUDA_ADD_LIBRARY(${name} ${type} ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
    INSTALL(TARGETS ${name} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR} 
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR} NAMELINK_SKIP
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR}  )
ENDMACRO(ADD_ELVIS_CUDA_LIBRARY name)

MACRO(ADD_ELVIS_OPTIX_LIBRARY name bindir)
    ADD_OPTIX_LIBRARY(${name} ${bindir} ${ARGN})
    SET_ELVIS_TARGET_PROPERTIES(${name})
ENDMACRO()

SET(PROJECTS 
  ElVis
  Utilities
)

ADD_SUBDIRECTORY(${PROJECTS})

if(APPLE)	
    set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
endif(APPLE)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/CMake/ElVisConfig.cmake.in 
               ${CMAKE_BINARY_DIR}/ElVisConfig.cmake @ONLY)

INSTALL(FILES ${CMAKE_BINARY_DIR}/ElVisConfig.cmake 
              DESTINATION ${CMAKE_INSTALL_PREFIX}) 

INSTALL(FILES ${CMAKE_SOURCE_DIR}/../ColorMaps/Grayscale.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/BlueToYellow.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/BlackBody.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/diverging257.cmap
              DESTINATION ${ELVIS_COLOR_MAP_DIR})

IF(WIN32)
    INSTALL(FILES ${PROJECT_SOURCE_DIR}/Externals/glut/release/glut64.dll
              DESTINATION ${ELVIS_BIN_DIR}) 
ENDIF()
# Set after the first configure pass is done, and can be used to set default values for 
# variables that users should be able to modify.
SET(PASSED_FIRST_CONFIGURE ON CACHE BOOL INTERNAL)
MARK_AS_ADVANCED(PASSED_FIRST_CONFIGURE)

###########################################
# Set RPATH information so the executable does not need to be run in the bin directory
###########################################

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)



#------------------------------------------------------------------------------
#Generate a unit tests target that runs all the unit tests in a single executable
#This must be the last thing generated
ADD_ELVIS_BOOST_TEST_OBJECTS( unit ALL_BOOST_UNIT_OBJECTS ElVisCore )


#Custom targets for debugging cmake
ADD_CUSTOM_TARGET( debug_cmake COMMAND ${CMAKE_COMMAND} -Wdev --warn-uninitialized WORKSPACE ${CMAKE_BINARY_DIR} )
ADD_CUSTOM_TARGET( debug_trace_cmake COMMAND ${CMAKE_COMMAND} -Wdev --warn-uninitialized --trace WORKSPACE ${CMAKE_BINARY_DIR} )
