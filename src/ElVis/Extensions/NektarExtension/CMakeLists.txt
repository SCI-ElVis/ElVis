SET(ProjectName "NektarExtension")

ENABLE_LANGUAGE(Fortran)
ADD_DEFINITIONS(-DNEKTAR_EXTENSION_EXPORTS)

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)

INCLUDE(../../Core/ElVisFiles.cmake)

SET(NEKTAR_EXTENSION_HEADERS
    Declspec.h
    NektarModel.h
    PluginInterface.h
) 

SET(NEKTAR_EXTENSION_SOURCES 
   NektarModel.cpp
   PluginInterface.cpp
)


SET(NEKTAR_EXTENSION_CUDA_SOURCE
    ExtensionInterface.cu
)


INCLUDE_DIRECTORIES(include )
#LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} ${NEKTAR++_LIBRARY_DIRS})

# Need to use ADD_OPTIX_LIBRARY so the .cu files get routed to nvcc.
ADD_OPTIX_LIBRARY(${ProjectName} ${ELVIS_BIN_DIR}
    ${NEKTAR_EXTENSION_SOURCES} 
    ${NEKTAR_EXTENSION_HEADERS} 
    ${NEKTAR_EXTENSION_CUDA_SOURCE}
    ${ELVIS_CUDA_SRC}
)

TARGET_LINK_LIBRARIES(${ProjectName} 
    ElVisCore
    veclib
    nektar3d
    hybrid
)


SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

INSTALL(TARGETS ${ProjectName} EXPORT ElVisLibraries RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  )



SUBDIRS(
  Veclib
  Hlib
  Nektar3d
  #NektarConverter
)
