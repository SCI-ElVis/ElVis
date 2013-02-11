
SET(ProjectName "ElVisModelConversion")

FIND_PACKAGE(Nektar++ REQUIRED)

IF( NEKTAR++_FOUND )
    SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${NEKTAR++_ROOT_DIR} ${CMAKE_MODULE_PATH})
    FIND_PACKAGE( Boost REQUIRED COMPONENTS thread iostreams zlib date_time filesystem system)

ENDIF()

SET(Headers
	DynamicLib.h
	IElVisModelConverter.h
) 

SET(Sources
	DynamicLib.cpp
	IElVisModelConverter.cpp
)


INCLUDE_DIRECTORIES(${NEKTAR++_INCLUDE_DIRS})

LINK_DIRECTORIES(${NEKTAR++_LIBRARY_DIRS})
ADD_ELVIS_STATIC_LIBRARY(${ProjectName} ${Headers}
	${Sources} )

TARGET_LINK_LIBRARIES(${ProjectName} 
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
)

SET_TARGET_PROPERTIES(${ProjectName} PROPERTIES VERSION ${ELVIS_VERSION})

INSTALL(FILES ${Headers} DESTINATION ${ELVIS_INCLUDE_DIR}/ElVisModelConversion)




