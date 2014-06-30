cmake_minimum_required(VERSION 2.8.11)
project(ElVis1)

# Disallow in-source build
if ("${CMAKE_SOURCE_DIR}"  STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR
    "ElVis requires an out of source Build. "
    "Please create a separate binary directory and run CMake there.")
endif()

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

#IF( NOT(CMAKE_INSTALL_PREFIX_SET) )
    SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE PATH "Install Path" FORCE)
#ENDIF()
#SET(CMAKE_INSTALL_PREFIX_SET ON CACHE INTERNAL "")

PROJECT(ElVis)

ENABLE_TESTING()
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_MODULE_PATH})

###########################################
# APPLICATION Information
###########################################
SET(APPLICATION_NAME "ElVis1" CACHE INTERNAL "Name of the application" FORCE)

###########################################
# SETUP DEFINTIONS FOR EXTERNAL PACKAGES
###########################################
IF( NOT WIN32 )
  # glew must be a DLL on windows.
  ADD_DEFINITIONS(-DGLEW_STATIC)
ENDIF()

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE AND NOT PASSED_FIRST_CONFIGURE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF()

SET(ELVIS_VERSION 0.1.1)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/bin ELVIS_BIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/cubin ELVIS_CUBIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/plugins ELVIS_PLUGIN_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/lib ELVIS_LIB_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/include ELVIS_INCLUDE_DIR)
FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX}/ColorMaps ELVIS_COLOR_MAP_DIR)

###########################################
# Find QT
# We need QT - We use a modified version of
# the FindQt4.cmake file that can be set
# to search for different versions of QT
# just point QT_DIR to the right directory
# to skip the use of the default installed
# QT distribution. 
###########################################a
if( MSVC )
  set (CMAKE_PREFIX_PATH "C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Lib")
endif()

find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Network REQUIRED)
find_package(Qt5WebKit REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL REQUIRED)

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${Qt5Network_INCLUDE_DIRS})
include_directories(${Qt5WebKit_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})

foreach(plugin ${Qt5Network_PLUGINS})
  get_target_property(_loc ${plugin} LOCATION)
  message("Plugin ${plugin} is at location ${_loc}")
endforeach()

#IF (QT_FOUND)
#  SET(QT_USE_QTOPENGL 1)
#  INCLUDE(${QT_USE_FILE})
#ELSE (QT_FOUND)
#  MESSAGE(FATAL_ERROR "QT ${QT_MIN_VERSION} or later is required for building Seg3D")
#ENDIF (QT_FOUND)

###########################################
# Instruct CMake to run moc automatically when needed.
###########################################
#set(CMAKE_AUTOMOC ON)

###########################################
# Instruct Qt to not use keywords like signals but
# rather Q_SIGNALS because it interferes with
# boost signals
###########################################
add_definitions(-DQT_NO_KEYWORDS)

###########################################
# External packages other than Qt
# Include the External packages, these are
# the packages that are shipped with 
# ElVis2 to lower the number of dependencies.
# Qt is currently not included as the code
# is a huge repository and to allow it to
# be build against different licensed 
# versions of Qt.
###########################################

include(FindCUDA.cmake)
include(SetupOptiX.cmake)

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
#INCLUDE(CheckComputeCapability.cmake)
#MESSAGE(STATUS "CUDA COMPUTE ${CUDA_COMPUTE_CAPABILITY}")

SET(PASSED_FIRST_CONFIGURE ON CACHE BOOL INTERNAL)

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
    ${GLEW_INCLUDE}
    ${GLUT_INCLUDE_DIR}
	${QT_INCLUDE_DIR}
	${BOOST_INCLUDE}
	${CMAKE_CURRENT_SOURCE_DIR}
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

MACRO(SET_ELVIS_TARGET_PROPERTIES name)

  IF (NOT ELVIS_TARGET_PROPERTIES_SET)
    SET(NOT ELVIS_TARGET_PROPERTIES_SET TRUE CACHE INTERNAL "Guards target properties so they are only set once" FORCE)
    SET_TARGET_PROPERTIES(${name} PROPERTIES DEBUG_POSTFIX -g)
    SET_TARGET_PROPERTIES(${name} PROPERTIES MINSIZEREL_POSTFIX -ms)
    SET_TARGET_PROPERTIES(${name} PROPERTIES RELWITHDEBINFO_POSTFIX -rg)

    SET(ADDITIONAL_COMPILE_FLAGS "")
    get_target_property(ADDITIONAL_COMPILE_FLAGS ${name} COMPILE_FLAGS)
    IF( ${ADDITIONAL_COMPILE_FLAGS} STREQUAL "ADDITIONAL_COMPILE_FLAGS-NOTFOUND" )
        SET(ADDITIONAL_COMPILE_FLAGS "")
    ENDIF()
    IF( CMAKE_COMPILER_IS_GNUCXX )
        # Some versions of gcc can't handle -g3, but it is set by a cuda
        # CMakeLists
        SET(STR1 " ")
        SET(STR2 " ")
        STRING(REPLACE "-g3" "" STR1 ${CMAKE_CXX_FLAGS_DEBUG} )
        STRING(REPLACE "-g" "" STR2 ${STR1} )
        SET(CMAKE_CXX_FLAGS_DEBUG "${STR2} -g" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  
        STRING(REPLACE "-g3" "" STR3 ${CMAKE_CXX_FLAGS_RELEASE} )
        STRING(REPLACE "-g" "" STR4 ${STR3} )
        SET(CMAKE_CXX_FLAGS_RELEASE "${STR4}" CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
    ENDIF()

    IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
        # The static libraries must be compiled with position independent code
        # on 64 bit Linux.
        SET(ADDITIONAL_COMPILE_FLAGS "${ADDITIONAL_COMPILE_FLAGS} -fPIC")
    ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
    
    IF( ADDITIONAL_COMPILE_FLAGS )
        SET_TARGET_PROPERTIES(${name} PROPERTIES COMPILE_FLAGS ${ADDITIONAL_COMPILE_FLAGS})
    ENDIF()

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
  ENDIF  (NOT ELVIS_TARGET_PROPERTIES_SET)
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

#ADD_SUBDIRECTORY(ElVis)
#ADD_SUBDIRECTORY(ElVis/Core)
#ADD_SUBDIRECTORY(ElVis/Extensions)

SET(PROJECTS 
  ElVis
  Utilities
)

SUBDIRS(${PROJECTS})

SET(CUDA_GENERATED_OUTPUT_DIR ${ELVIS_CUBIN_DIR} CACHE PATH "" FORCE)

IF( MSVC )	
    # Removes the warnings about unsafe methods such as strcpy, std::copy, memcmp, etc.
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_DEPRECATE)

    # Adds support for M_PI
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)	
ENDIF( MSVC )

###########################################
#Add path for glu32 for windows machines
###########################################
IF(WIN32)
	set (CMAKE_LIBRARY_PATH "C:\\Program Files (x86)\\Windows Kits\\8.0\\Lib\\win8\\um\\x64")
ENDIF()


###########################################
# Find includes in corresponding build directories
###########################################
#set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(APPLE)	
    set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
endif(APPLE)

SET(CONFIGURE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Configuration/ CACHE INTERNAL "Path to the configuration directory" FORCE)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVisConfig.cmake.in 
               ${CMAKE_BINARY_DIR}/ElVisConfig.cmake @ONLY)

INSTALL(FILES ${CMAKE_BINARY_DIR}/ElVisConfig.cmake 
              DESTINATION ${CMAKE_INSTALL_PREFIX}) 

INSTALL(FILES ${CMAKE_SOURCE_DIR}/../ColorMaps/Grayscale.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/BlueToYellow.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/BlackBody.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/Rainbow.xml
              ${CMAKE_SOURCE_DIR}/../ColorMaps/diverging257.cmap
              DESTINATION ${ELVIS_COLOR_MAP_DIR})

if(WIN32)
    INSTALL(FILES ${PROJECT_SOURCE_DIR}/Externals/glut/release/glut64.dll
              DESTINATION ${ELVIS_BIN_DIR}) 
endif()
# Set after the first configure pass is done, and can be used to set default values for 
# variables that users should be able to modify.
SET(PASSED_FIRST_CONFIGURE ON CACHE BOOL INTERNAL)
MARK_AS_ADVANCED(PASSED_FIRST_CONFIGURE)

SET(CUDA_GENERATED_OUTPUT_DIR ${ELVIS_CUBIN_DIR} CACHE PATH "" FORCE)

