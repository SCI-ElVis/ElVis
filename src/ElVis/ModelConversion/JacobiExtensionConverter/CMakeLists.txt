
SET(ProjectName IsosurfaceConversionModule)

SET(Headers
	IsosurfaceConverter.h
	IsosurfaceConverterDeclspec.h
)

SET(Sources
	IsosurfaceConverter.cpp
)

ADD_DEFINITIONS(-DISOSURFACE_CONVERTER_EXPORTS)

INCLUDE_DIRECTORIES(${ELVIS_INCLUDE_DIR} ${Boost_INCLUDE_DIR})

ADD_ELVIS_PLUGIN(${ProjectName} ${Headers} ${Sources})
TARGET_LINK_LIBRARIES(${ProjectName} 
	${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY}
	OriginalNektar
	ElVisModelConversion
)
