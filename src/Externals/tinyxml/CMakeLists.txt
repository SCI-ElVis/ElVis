###########################################
# TinyXML project

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(TinyXML)
SET(TinyXML_Library TinyXML CACHE INTERNAL "Boost library name." FORCE)

SET(TINYXML_SOURCES
    tinyxml.cpp
    tinyxmlparser.cpp 
    tinyxmlerror.cpp 
    tinystr.cpp
)

SET(TINYXML_HEADERS
    tinystr.h
    tinyxml.h
)

SET(TINY_XML_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "TinyXML include directory" FORCE)
ADD_LIBRARY(${TinyXML_Library} STATIC ${TINYXML_SOURCES} ${TINYXML_HEADERS})
SET_PROPERTY(TARGET ${TinyXML_Library} PROPERTY FOLDER "Externals")
