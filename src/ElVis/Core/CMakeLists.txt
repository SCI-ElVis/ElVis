
SET(ProjectName ElVisCore)

FIND_PACKAGE(OpenGL REQUIRED)
# Default location
IF( MSVC )
    SET(GLUT_ROOT_PATH ${CMAKE_SOURCE_DIR}/Externals/glut)
    SET(GLUT_INCLUDE_DIR ${GLUT_ROOT_PATH}/include)
    SET(GLUT_glut_LIBRARY ${GLUT_ROOT_PATH}/release/glut32.lib)
ENDIF()

IF (APPLE) #set the env vars to use the cuda path
    SET(ENV_PATH $ENV{PATH})
    SET(ENV_DYLD $ENV{DYLD_LIBRARY_PATH})
    SET(ENV{PATH} ${CUDA_TOOLKIT_ROOT_DIR}/bin)
    SET(ENV{DYLD_LIBRARY_PATH} ${CUDA_TOOLKIT_ROOT_DIR}/lib)
ENDIF()

#############################
# Check for GPUs present and their compute capability
# based on http://stackoverflow.com/questions/2285185/easiest-way-to-test-for-existence-of-cuda-capable-gpu-from-cmake/2297877#2297877 (Christopher Bruns)
if(CUDA_FOUND AND NOT CUDA_HAVE_GPU)
    message(STATUS "Checking CUDA compute capability.")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR} 
        ${CMAKE_SOURCE_DIR}/cuda_compute_capability.c
        CMAKE_FLAGS 
        -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
        -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
        RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)
    #message(STATUS "Compile: ${RUN_OUTPUT_VAR}")
    #if (COMPILE_RESULT_VAR)
    #    message(STATUS "compiled -> " ${RUN_RESULT_VAR})
    #else()
    #    message(STATUS "didn't compile")
    #endif()
    # COMPILE_RESULT_VAR is TRUE when compile succeeds
    # RUN_RESULT_VAR is zero when a GPU is found
    if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
        #message(STATUS "worked")
        set(CUDA_HAVE_GPU TRUE CACHE BOOL "Whether CUDA-capable GPU is present")
        set(CUDA_COMPUTE_CAPABILITY ${RUN_OUTPUT_VAR} CACHE STRING "Compute capability of CUDA-capable GPU present")
        set(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}" CACHE STRING "Which GPU architectures to generate code for (each arch/code pair will be passed as --generate-code option to nvcc, separate multiple pairs by ;)")
        mark_as_advanced(CUDA_COMPUTE_CAPABILITY CUDA_GENERATE_CODE)
        LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21" "-arch=sm_30" "-arch=sm_31")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_${CUDA_COMPUTE_CAPABILITY}"  CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
        SET(CUDA_COMPUTE_CAPABILITY_SUCCESS TRUE CACHE BOOL "")
    else()
        #message(STATUS "didn't work")
        SET(CUDA_COMPUTE_CAPABILITY_SUCCESS FALSE CACHE BOOL "")
        set(CUDA_HAVE_GPU FALSE CACHE BOOL "Whether CUDA-capable GPU is present")
    endif()
endif()

IF (NOT CUDA_COMPUTE_CAPABILITY_SUCCESS)
    SET(CUDA_USE_COMPUTE_CAPABILITY_10 false CACHE BOOL "Use Compute Capability 1.0")
    SET(CUDA_USE_COMPUTE_CAPABILITY_11 false CACHE BOOL "Use Compute Capability 1.1")
    SET(CUDA_USE_COMPUTE_CAPABILITY_12 false CACHE BOOL "Use Compute Capability 1.2")  
    SET(CUDA_USE_COMPUTE_CAPABILITY_13 false CACHE BOOL "Use Compute Capability 1.3")
    SET(CUDA_USE_COMPUTE_CAPABILITY_20 true CACHE BOOL "Use Compute Capability 2.0")
    SET(CUDA_USE_COMPUTE_CAPABILITY_21 false CACHE BOOL "Use Compute Capability 2.1")
    
    IF(CUDA_USE_COMPUTE_CAPABILITY_10)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_10")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ELSEIF(CUDA_USE_COMPUTE_CAPABILITY_11)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_11")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ELSEIF(CUDA_USE_COMPUTE_CAPABILITY_12)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_12")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ELSEIF(CUDA_USE_COMPUTE_CAPABILITY_13)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_13")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ELSEIF(CUDA_USE_COMPUTE_CAPABILITY_20)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ELSEIF(CUDA_USE_COMPUTE_CAPABILITY_21)
      LIST(REMOVE_ITEM CUDA_NVCC_FLAGS "-arch=sm_10" "-arch=sm_11" "-arch=sm_12" "-arch=sm_13" "-arch=sm_20" "-arch=sm_21")
      LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_21")
      SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE LIST "Semi-colon delimit multiple arguments." FORCE)
    ENDIF()
ENDIF ()

#set the env vars back
IF (APPLE)
    SET(ENV{PATH} ${ENV_PATH})
    SET(ENV{DYLD_LIBRARY_PATH} ${ENV_DYLD})
ENDIF ()

INCLUDE_DIRECTORIES(
    ${OPENGL_INCLUDE_DIR} 
    ${GLUT_INCLUDE_DIR}
  ${TINY_XML_DIR}
)
INCLUDE_DIRECTORIES(${ELVIS_CUDA_INCLUDE})                    
ADD_DEFINITIONS(-DELVIS_EXPORTS)
add_definitions(-DQT_NO_KEYWORDS)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisConfig.cpp.in
               ${CMAKE_BINARY_DIR}/ElVis/Core/ElVisConfig.cpp @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/Float.h.in
               ${CMAKE_BINARY_DIR}/ElVis/Core/Float.h @ONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVis/Core/Float.cu.in
               ${CMAKE_BINARY_DIR}/ElVis/Core/Float.cu )

SET(ELVIS_HEADERS
    Camera.h
    Color.h
    ColorMap.h
    Cuda.h
    CutSurfaceContourModule.h
    CutSurfaceMeshModule.h
    Cylinder.h
    DirectionalLight.h
    DynamicLib.h
    ElementId.h
    ElVisConfig.h
    ElVisConfig.cpp.in
    FaceDef.h
    FaceObject.h
    ${CMAKE_BINARY_DIR}/ElVis/Core/Float.h
    Float.h.in
    FieldInfo.h
    ElVisDeclspec.h
    ExtensionPTXConfig.cpp.in
    ExtensionPTXConfig.h.in
    HostTransferFunction.h
    Interval.hpp
    Jacobi.hpp
    Light.h
    Model.h
    Octree.hpp
    OpenGL.h
    OptiXBuffer.hpp
    OptiXExtensions.hpp
    Plugin.h
    Point.hpp
    PointLight.h
    PtxManager.h
    RayGeneratorProgram.h
    ReferencePointParameter.h
    Scene.h
    SceneView.h
    SceneViewProjection.h
    Spaces.h
    Stat.h
    sutil.h
    SynchedObject.hpp
    Timer.h
    TransferFunction.h
    Triangle.h
    Util.hpp
    Vector.hpp
    VolumeRenderingIntegrationCategory.h
) 

SET(ELVIS_MATH_HEADERS
    ../Math/AdaptiveQuadrature.hpp
    ../Math/GaussLegendreQuadrature.hpp
    ../Math/Integration.hpp
    ../Math/Jacobi.hpp
    ../Math/Polynomial.hpp
    ../Math/PolynomialProjection.hpp
    ../Math/TrapezoidalIntegration.hpp
)

SET(ELVIS_SOURCES
    Camera.cpp
    Color.cpp
    ColorMap.cpp
    CutSurfaceContourModule.cpp
    CutSurfaceMeshModule.cpp
    Cylinder.cpp
    DirectionalLight.cpp
    DynamicLib.cpp
    ${CMAKE_BINARY_DIR}/ElVis/Core/ElVisConfig.cpp
    FaceObject.cpp
    Light.cpp
    Model.cpp
    Plugin.cpp
    PointLight.cpp
    PtxManager.cpp
    Scene.cpp
    SceneView.cpp
    Timer.cpp
    TransferFunction.cpp
    Triangle.cpp
    Util.cpp
)

SET(ObjectFiles
    Object.h
    Object.cpp
    Plane.h
    Plane.cpp
    TwoDPrimaryElements.h
    TwoDPrimaryElements.cpp
)

SET(PrimaryRayObjectFiles
    PrimaryRayObject.h
    PrimaryRayObject.cpp
    SampleFaceObject.h
    SampleFaceObject.cpp
    SampleVolumeSamplerObject.h
    SampleVolumeSamplerObject.cpp
    SurfaceObject.h
    SurfaceObject.cpp
    TwoDPrimaryElementsPrimaryObject.h
    TwoDPrimaryElementsPrimaryObject.cpp
)

SET(ModuleFiles
    ColorMapperModule.h
    ColorMapperModule.cpp
    IsosurfaceModule.h
    IsosurfaceModule.cpp
    LightingModule.h
    LightingModule.cpp
    PrimaryRayModule.h
    PrimaryRayModule.cpp
    RenderModule.h
    RenderModule.cpp
    SampleOntoNrrdModule.h
    SampleOntoNrrdModule.cpp
    VolumeRenderingModule.h
    VolumeRenderingModule.cpp
)

SET(ELVIS_CUDA_HEADERS
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ConvertToColor.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/CutCylinder.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/CutSurfaceContourModule.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/CutSurfacePayloads.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/DiffuseLighting.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ElementTraversal.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisCuda.cu.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/FieldEvaluator.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/FindElement.cu
    ${CMAKE_BINARY_DIR}/ElVis/Core/Float.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/Float.cu.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/GaussKronrod.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/InnerIntegralAdaptiveTrapezoidal.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/IntervalMatrix.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/IntervalPoint.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/IsosurfaceOptiXModule.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/IsosurfaceModuleCuda.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/jacobi.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/matrix.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/OptixVariables.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/OuterIntegralAdaptiveTrapezoidal.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/Plane.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/PrimaryRayGenerator.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/PrimaryRayModule.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/Printf.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/OpenGLLighting.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ReentrantAdaptiveTrapezoidal.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/SampleVolumeSamplerObject.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/SurfaceObject.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/TextureColorMap.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/triangle.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/typedefs.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/util.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRendering.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRenderingFull.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRenderingOptiXModule.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRenderingSingleRayPerSegment.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRenderingOptiXModule.cu
    ${CMAKE_SOURCE_DIR}/ElVis/Core/VolumeRenderingPayload.cu
) 

INCLUDE_DIRECTORIES(
    ${CUDA_TOOLKIT_INCLUDE}
    ${PNG_INCLUDE}
    ${ZLIB_INCLUDE}
    ${CMAKE_SOURCE_DIR}/Externals/tinyxml
)

add_definitions(${PNG_DEFINITIONS})

SOURCE_GROUP("Cuda" FILES ${ELVIS_CUDA_HEADERS})
SET_SOURCE_FILES_PROPERTIES(${ELVIS_CUDA_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)
SOURCE_GROUP("Modules" FILES ${ModuleFiles})
SOURCE_GROUP("Primary Ray Objects" FILES ${PrimaryRayObjectFiles})
SOURCE_GROUP("Objects" FILES ${ObjectFiles})
SOURCE_GROUP("Core" FILES ${ELVIS_SOURCES} ${ELVIS_HEADERS})
ADD_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR} ${ELVIS_HEADERS} ${PrimaryRayObjectFiles} ${ObjectFiles} ${ELVIS_MATH_HEADERS} ${ELVIS_SOURCES} ${ModuleFiles} ${ELVIS_CUDA_HEADERS})

SET_PROPERTY(TARGET ${ProjectName} PROPERTY FOLDER "Core")

###########################################
# Specific defines that need to be set
# STATIC - we want glew to be a static library

#ADD_DEFINITIONS(-DGLEW_STATIC)

TARGET_LINK_LIBRARIES(${ProjectName} 
    ${BOOST_LIBRARY}
    ${GLEW_LIBRARY}
    ${GLUT_glut_LIBRARY}
    ${PNG_LIBRARY}
    ${TinyXML_Library}
    ${optix_LIBRARY} ${optixu_LIBRARY}
)


INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR} 
                                LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                                ARCHIVE DESTINATION ${ELVIS_LIB_DIR} )

#if(APPLE)	
#  set(CMAKE_SHARED_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
#endif(APPLE)
                
INSTALL(FILES ${optix_DLL} ${optixu_DLL} ${optix_LIBRARY} ${optixu_LIBRARY}
    Nodes.txt Weights.txt MonomialConversionTables.txt
    DESTINATION ${ELVIS_BIN_DIR}
)

INSTALL(FILES ${ELVIS_HEADERS} ${ELVIS_CUDA_HEADERS}
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.cpp.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ExtensionPTXConfig.h.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisOptiX.cu.in
    ${CMAKE_SOURCE_DIR}/ElVis/Core/ElVisCuda.cu.in
    DESTINATION ${ELVIS_INCLUDE_DIR}/ElVis/Core)

INSTALL(FILES ${ELVIS_MATH_HEADERS}
    DESTINATION ${ELVIS_INCLUDE_DIR}/ElVis/Math)



CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/ElVisConfig.cmake.in
               ${CMAKE_BINARY_DIR}/ElVisConfig.cmake @ONLY)
			   
SET (CORE_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "core build path")

MESSAGE(STATUS "CORE: ${CUDA_GENERATE_CODE}")
