
SET(ProjectName FunctionProjectionModule)

FIND_PACKAGE(Nektar++ REQUIRED)

IF( NEKTAR++_FOUND )
    SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${NEKTAR++_ROOT_DIR} ${CMAKE_MODULE_PATH})
    FIND_PACKAGE( Boost REQUIRED COMPONENTS thread iostreams zlib date_time filesystem system)

ENDIF()

SET(Headers
	FunctionProjection.h
	FunctionProjectionDeclspec.h
)

SET(Sources
	FunctionProjection.cpp
)

ADD_DEFINITIONS(-DFUNCTION_PROJECTION_EXPORTS)

INCLUDE_DIRECTORIES(${ELVIS_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${NEKTAR++_INCLUDE_DIRS} ${LOKI_INCLUDE_DIR})
LINK_DIRECTORIES(${NEKTAR++_LIBRARY_DIRS})

ADD_ELVIS_PLUGIN(${ProjectName} ${Headers} ${Sources})
TARGET_LINK_LIBRARIES(${ProjectName} 
	${Boost_LIBRARIES}
	ElVisModelConversion
	ElVisCore
    NektarPlusPlusExtension
	${NEKTAR++_LIBRARIES}
)

INSTALL(TARGETS ${ProjectName} RUNTIME DESTINATION ${ELVIS_BIN_DIR}
                               LIBRARY DESTINATION ${ELVIS_BIN_DIR}
                               ARCHIVE DESTINATION ${ELVIS_BIN_DIR}  )

