SET(ProjectName "NektarPlusPlusExtension")
SET(ExtensionPath ElVis/Extensions/${ProjectName})

FIND_PACKAGE(Nektar++ REQUIRED NO_MODULE NO_DEFAULT_PATH NO_CMAKE_BUILDS_PATH NO_CMAKE_PACKAGE_REGISTRY)

IF( NEKTAR++_FOUND )
    SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${NEKTAR++_ROOT_DIR} ${CMAKE_MODULE_PATH})
ENDIF()

ADD_DEFINITIONS(${NEKTAR++_DEFINITIONS})

# Configure the extension file for the Nektar++ extension.  Since Cuda required
# all code to be available at compile time, this step creates an extension
# specific version of these files that can then be compiled into an extension
# specific version of ElVis.
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.cpp @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.h @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu @ONLY)

#################################3
# Nektar++ Extension
#################################3
SET(NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY
  ExtensionOptiXInterface.cu
)
SET_SOURCE_FILES_PROPERTIES(
    ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY}
    PROPERTIES HEADER_FILE_ONLY ON)

SET(NEKTAR_PLUS_PLUS_EXTENSION_SOURCES
  ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.cpp
  ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.h
  Declspec.h
  NektarModel.cpp
  NektarModel.h
  PluginInterface.cpp
  PluginInterface.h
  ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
) 

INCLUDE_DIRECTORIES(${NEKTAR++_INCLUDE_DIRS} ${NEKTAR++_TP_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} ${NEKTAR++_LIBRARY_DIRS} ${NEKTAR++_TP_LIBRARY_DIRS})

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_ELVIS_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${NEKTAR_PLUS_PLUS_EXTENSION_SOURCES} 
    ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY}
)

TARGET_LINK_LIBRARIES(${ProjectName} 
  ${NEKTAR++_LIBRARIES}
  ElVisCore
)

source_group("" FILES ${NEKTAR_PLUS_PLUS_EXTENSION_SOURCES} ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY})
SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Extensions")

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries 
  RUNTIME DESTINATION ${ELVIS_BIN_DIR}
  LIBRARY DESTINATION ${ELVIS_BIN_DIR}
  ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  
)
