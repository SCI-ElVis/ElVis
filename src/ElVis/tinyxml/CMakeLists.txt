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

ADD_ELVIS_LIBRARY(tinyxml STATIC ${TINYXML_SOURCES} ${TINYXML_HEADERS})
