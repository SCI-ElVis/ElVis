SET(ProjectName "ProjectXExtension")
SET(ExtensionPath ElVis/Extensions/${ProjectName})

SET(ProjectX_DIR ${CMAKE_SOURCE_DIR}/ElVis/Extensions/ProjectXExtension)
SET(LEDA_DIR ${CMAKE_SOURCE_DIR}/ElVis/Extensions/ProjectXExtension)

FIND_PACKAGE(ProjectX)
FIND_PACKAGE(LEDA)
FIND_PACKAGE( LAPACK REQUIRED )

IF(PROJECTX_FOUND)
  INCLUDE_DIRECTORIES( ${PROJECTX_INCLUDE_DIR} ${PROJECTX_INCLUDE_DIR}/Fundamentals ${PROJECTX_INCLUDE_DIR}/Reference ${PROJECTX_INCLUDE_DIR}/Grid ${PROJECTX_INCLUDE_DIR}/Grid/CutCell3D ${PROJECTX_INCLUDE_DIR}/Compressible ${LEDA_INCLUDE_DIR})
ENDIF(PROJECTX_FOUND)

ADD_DEFINITIONS(-DELVIS_USE_PROJECTX)

SET(ProjectX_PLUGIN_DIR ${ELVIS_PLUGIN_DIR}/ProjectXExtension)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/ProjectXExtension/ProjectXExtensionPTXConfig.cpp
               @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/ProjectXExtension/ProjectXExtensionPTXConfig.h
               @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
               @ONLY)
#               CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisCuda.cu.in
#               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisCuda.cu
#               @ONLY)

#################################
# PX Extension Source
#################################
SET(ProjectX_EXTENSION_HEADERS PXModel.h PXStructDefinitions.h)

SET(ProjectX_EXTENSION_SOURCES PXModel.cpp )

SET(ProjectX_EXTENSION_ELVIS_HEADERS ${CMAKE_BINARY_DIR}/ElVis/Extensions/ProjectXExtension/ProjectXExtensionPTXConfig.h
    )

SET(ProjectX_EXTENSION_ELVIS_SOURCES ${CMAKE_BINARY_DIR}/ElVis/Extensions/ProjectXExtension/ProjectXExtensionPTXConfig.cpp
    )


#################################
# PX OptiX Source
#################################
SET(ProjectX_OPTIX_HEADER_ONLY
    ExtensionOptiXInterface.cu
    PXSimplex.cu
    PXShape_Elvis.cu
    PXNormal_Elvis.cu
    PXQuadReference_Elvis.cu
    PXCutCell_ElVis.cu
    PXCoordinates_Elvis.cu
    PXOptiXCudaCommon.cu
)

SET_SOURCE_FILES_PROPERTIES(${ProjectX_OPTIX_HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY ON)

SET(ProjectX_Extension_OptiX ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
    )

SOURCE_GROUP("OptiX" FILES ${ProjectX_OPTIX_HEADER_ONLY} ${ProjectX_Extension_OptiX})


#################################3
# Create Libraries
#################################3

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_ELVIS_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${ProjectX_EXTENSION_HEADERS}
    ${ProjectX_EXTENSION_SOURCES}
    ${ProjectX_EXTENSION_ELVIS_HEADERS}
    ${ProjectX_EXTENSION_ELVIS_SOURCES}
    ${ELVIS_CUDA}
    ${ProjectX_OPTIX_HEADER_ONLY}
    ${ProjectX_Extension_OptiX}
)


TARGET_LINK_LIBRARIES(${ProjectName} ElVisCore )

TARGET_LINK_LIBRARIES(${ProjectName}
  ${PROJECTX_LIBRARIES}
  ${LEDA_LIBRARIES}
  ${LAPACK_LIBRARIES}
  )

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION
${ELVIS_VERSION})

INSTALL(FILES ${ProjectX_EXTENSION_HEADERS} DESTINATION
 ${ELVIS_INCLUDE_DIR}/ProjectXExtension)

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR} )

