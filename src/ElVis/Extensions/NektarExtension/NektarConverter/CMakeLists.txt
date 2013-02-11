
SET(ProjectName NektarConversionModule)

SET(Headers
	NektarConverter.h
	NektarConverterDeclspec.h
)

SET(Sources
	NektarConverter.cpp
)

ADD_DEFINITIONS(-DNEKTAR_CONVERTER_EXPORTS)

INCLUDE_DIRECTORIES(${ELVIS_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/Modules/NektarConversion/include)

ADD_ELVIS_PLUGIN(${ProjectName} ${Headers} ${Sources})
TARGET_LINK_LIBRARIES(${ProjectName} 
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	OriginalNektar
	ElVisModelConversion
	nektar3d
    hybrid
    veclib
)
