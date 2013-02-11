#SUBDIRS(
    #Converter
    #Interface
#)

#IF( CMAKE_HOST_UNIX )   
    #SUBDIRS(NektarConversion)
#ENDIF()

SET(ELVIS_ENABLE_FUNCTION_PROJECTION_MODULE OFF CACHE BOOL "Enable the function projection module for Nektar++.")
SET(ELVIS_ENABLE_DATA_CONVERTER OFF CACHE BOOL "Enable module that converts data to Nektar++ format.")

IF( ELVIS_ENABLE_FUNCTION_PROJECTION_MODULE )
	SUBDIRS(FunctionProjection)
ENDIF()

IF( ELVIS_ENABLE_DATA_CONVERTER )
	SUBDIRS(Interface Converter)
ENDIF()