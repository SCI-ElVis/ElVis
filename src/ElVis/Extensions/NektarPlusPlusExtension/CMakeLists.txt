SET(ProjectName "NektarPlusPlusExtension")

ADD_DEFINITIONS(-DNEKTAR_PLUS_PLUS_EXTENSION_EXPORTS -DNEKTAR_USING_BLAS -DNEKTAR_USING_LAPACK)

FIND_PACKAGE(Nektar++ REQUIRED NO_MODULE NO_DEFAULT_PATH NO_CMAKE_BUILDS_PATH NO_CMAKE_PACKAGE_REGISTRY)

IF( NEKTAR++_FOUND )
    SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${NEKTAR++_ROOT_DIR} ${CMAKE_MODULE_PATH})
ENDIF()

# Configure the extension file for the Nektar++ extension.  Since Cuda required all code to be 
# available at compile time, this step creates an extension specific version of these files that
# can then be compiled into an extension specific version of ElVis.
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.cpp @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.h @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisCuda.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisCuda.cu @ONLY)

#################################3
# Nektar++ Optix Source
#################################3
SET(NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY
    ExtensionOptiXInterface.cu
    Expansions.cu
    NektarModel.cu
    OptixHexahedron.cu
    typedefs.cu
)
SET_SOURCE_FILES_PROPERTIES(${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY ON)

SET(NEKTAR_PLUS_PLUS_EXTENSION_OPTIX
    ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
)
#SOURCE_GROUP("OptiX" FILES ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX} ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY})


#################################3
# Nektar++ Cuda Source
#################################3
SET(NEKTAR_PLUS_PLUS_EXTENSION_CUDA_HEADER_ONLY
    CudaHexahedron.cu
    ExtensionCudaInterface.cu
    Expansions.cu
)
SET_SOURCE_FILES_PROPERTIES(${NEKTAR_PLUS_PLUS_EXTENSION_CUDA_HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY ON)
SET(NEKTAR_PLUS_PLUS_EXTENSION_CUDA
    ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisCuda.cu
)


#################################3
# Nektar++ Extension
#################################3
SET(NEKTAR_PLUS_PLUS_EXTENSION_HEADERS
    Declspec.h
    NektarModel.h
    PluginInterface.h
    ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.h
) 

SET(NEKTAR_PLUS_PLUS_EXTENSION_SOURCES 
   NektarModel.cpp
   PluginInterface.cpp
   ${CMAKE_BINARY_DIR}/ElVis/Extensions/NektarPlusPlusExtension/NektarPlusPlusExtensionPTXConfig.cpp
)

INCLUDE_DIRECTORIES(${NEKTAR++_INCLUDE_DIRS} )
LINK_DIRECTORIES(${BOOST_LIBRARY} ${NEKTAR++_LIBRARY_DIRS})

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_ELVIS_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${NEKTAR_PLUS_PLUS_EXTENSION_SOURCES} 
    ${NEKTAR_PLUS_PLUS_EXTENSION_HEADERS} 
    ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEADER_ONLY}
    ${NEKTAR_PLUS_PLUS_EXTENSION_OPTIX}
)

TARGET_LINK_LIBRARIES(${ProjectName} 
    ${NEKTAR++_LIBRARIES}
    #tinyxml
	${SCI_TINYXML_LIBRARY}
    ${BOOST_LIBRARY}
    ElVisCore
)

ADD_ELVIS_CUDA_LIBRARY(${ProjectName}Cuda 
    ${NEKTAR_PLUS_PLUS_EXTENSION_CUDA_HEADER_ONLY}
    ${NEKTAR_PLUS_PLUS_EXTENSION_CUDA}
)

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  )

