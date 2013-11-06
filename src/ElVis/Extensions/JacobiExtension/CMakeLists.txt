SET(ProjectName "JacobiExtension")
SET(ExtensionPath ElVis/Extensions/${ProjectName})
ADD_DEFINITIONS(-DJACOBI_EXTENSION_EXPORTS)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.cpp @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in 
               ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.h @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu @ONLY)

SET(JACOBI_EXTENSION_SOURCES 
  ${CMAKE_BINARY_DIR}/ElVis/${ProjectName}/Core/ElVisOptiX.cu
  ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.cpp
  ${CMAKE_BINARY_DIR}/ElVis/Extensions/JacobiExtension/JacobiExtensionPTXConfig.h
  CastAndFilterIterator.hpp
  Common.cu
  Declspec.h
  Edge.cpp 
  Edge.h
  EndianConvert.h
  ExtensionOptiXInterface.cu
  Face.cpp
  Face.h
  FiniteElementMath.cpp   
  FiniteElementMath.h
  FiniteElementVolume.cpp   
  FiniteElementVolume.h
  Hexahedron.cpp   
  Hexahedron.h
  HexahedronCommon.cu
  IPolyhedron.h
  Isosurface.cpp            
  Isosurface.h
  Jacobi.hpp
  JacobiExtensionElVisModel.cpp
  JacobiExtensionElVisModel.h
  NumericalIntegration.h
  NumericalIntegration.hpp
  OptiXHexahedron.cu
  OptiXPrism.cu
  PluginInterface.cpp  
  PluginInterface.h
  Point.hpp
  PointTransformations.cpp  
  PointTransformations.hpp
  Polyhedra.cpp             
  Polyhedra.h
  PolyhedraBoundingBox.cpp  
  PolyhedraBoundingBox.h
  Polylib.cpp
  Polylib.h
  Polynomial.cpp
  Polynomial.h
  Polynomial.hpp
  PolynomialInterpolation.h
  PolynomialInterpolation.hpp
  Prism.cpp
  Prism.h
  PrismCommon.cu
  Pyramid.cpp
  Pyramid.h
  RayTracerDialog.cpp
  SimpleMatrix.hpp
  Spaces.h
  Tetrahedron.cpp
  Tetrahedron.h
  Typedefs.h
  Vector.hpp
  Writers.cpp
  Writers.h
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
source_group("" FILES ${JACOBI_EXTENSION_SOURCES} ${JACOBI_EXTENSION_OPTIX_HEADER_ONLY})

#################################3
# Create Libraries
#################################3

LINK_DIRECTORIES(
    ${Boost_LIBRARY_DIRS}
)
#ADD_DEFINITIONS(-DGLEW_STATIC)

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_ELVIS_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${JACOBI_EXTENSION_SOURCES} 
    ${JACOBI_EXTENSION_OPTIX_HEADER_ONLY}
)


TARGET_LINK_LIBRARIES(${ProjectName} 
    ElVisCore
    glew
)

SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Extensions")

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  )

