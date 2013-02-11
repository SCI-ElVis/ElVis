SET(ProjectName "JacobiExtension")

ADD_DEFINITIONS(-DJACOBI_EXTENSION_EXPORTS)


SET(JACOBI_PLUGIN_DIR ${ELVIS_PLUGIN_DIR}/ElVisJacobi)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.cpp @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.h @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisCuda.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisCuda.cu @ONLY)

SET(JACOBI_EXTENSION_HEADERS
    CastAndFilterIterator.hpp
    Jacobi.hpp
    NumericalIntegration.hpp
    PluginInterface.h
    PointTransformations.hpp
    Polynomial.hpp
    PolynomialInterpolation.hpp
    SimpleMatrix.hpp
    Declspec.h
    Edge.h
    EndianConvert.h
    Face.h
    FiniteElementMath.h
    FiniteElementVolume.h
    Hexahedron.h
    IPolyhedron.h
    Isosurface.h
    NumericalIntegration.h
    Polyhedra.h
    PolyhedraBoundingBox.h
    Polylib.h
    Polynomial.h
    PolynomialInterpolation.h
    Prism.h
    Pyramid.h
    Tetrahedron.h
    Typedefs.h
    Writers.h
    Point.hpp
    Spaces.h
    Vector.hpp
) 

SET(JACOBI_EXTENSION_SOURCES 
    Edge.cpp 
    Polylib.cpp
    Face.cpp
    FiniteElementMath.cpp   
    PluginInterface.cpp  
    Polynomial.cpp
    FiniteElementVolume.cpp   
    Prism.cpp
    Hexahedron.cpp   
    Pyramid.cpp
    Isosurface.cpp            
    RayTracerDialog.cpp
    Tetrahedron.cpp
    PointTransformations.cpp  
    Writers.cpp
    Polyhedra.cpp             
    PolyhedraBoundingBox.cpp  
)

#################################3
# Jacobi Extension Source
#################################3
SET(JACOBI_EXTENSION_ELVIS_HEADERS
    JacobiExtensionElVisModel.h
    ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.h
)

SET(JACOBI_EXTENSION_ELVIS_SOURCES
    JacobiExtensionElVisModel.cpp
    ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.cpp
)

#################################3
# Jacobi Optix Source
#################################3
SET(JACOBI_EXTENSION_OPTIX_HEADER_ONLY
    Common.cu
    ExtensionOptiXInterface.cu
    HexahedronCommon.cu
    OptiXHexahedron.cu
    OptiXPrism.cu
    PrismCommon.cu
)
SET_SOURCE_FILES_PROPERTIES(${JACOBI_EXTENSION_OPTIX_HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY ON)

SET(JACOBI_EXTENSION_OPTIX
    ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
)
SOURCE_GROUP("OptiX" FILES ${JACOBI_EXTENSION_OPTIX} ${JACOBI_EXTENSION_OPTIX_HEADER_ONLY})




#################################3
# Jacobi Cuda Source
#################################3
SET(JACOBI_EXTENSION_CUDA_HEADER_ONLY
    CudaHexahedron.cu
    CudaPrism.cu
    HexahedronCommon.cu
    ExtensionCudaInterface.cu
    PrismCommon.cu
)
SET_SOURCE_FILES_PROPERTIES(${JACOBI_EXTENSION_CUDA_HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY ON)
SET(JACOBI_EXTENSION_CUDA
    ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisCuda.cu
)


#################################3
# Create Libraries
#################################3

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_ELVIS_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${JACOBI_EXTENSION_HEADERS} 
    ${JACOBI_EXTENSION_SOURCES} 
    ${JACOBI_EXTENSION_CUDA_SOURCE}
    ${JACOBI_EXTENSION_ELVIS_HEADERS}
    ${JACOBI_EXTENSION_ELVIS_SOURCES}
    ${JACOBI_EXTENSION_OPTIX}
    ${JACOBI_EXTENSION_OPTIX_HEADER_ONLY}
)

ADD_ELVIS_CUDA_LIBRARY(${ProjectName}Cuda 
    ${JACOBI_EXTENSION_CUDA_HEADER_ONLY}
    ${JACOBI_EXTENSION_CUDA}
)

TARGET_LINK_LIBRARIES(${ProjectName} 
    ElVisCore
)


SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

#INSTALL(FILES ${JACOBI_EXTENSION_HEADERS} DESTINATION ${ELVIS_INCLUDE_DIR}/JacobiExtension)

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  )

