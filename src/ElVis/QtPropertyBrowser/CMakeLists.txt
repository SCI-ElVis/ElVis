SET(Project "QtPropertyBrowser")

IF( NOT QT4_FOUND )
    FIND_PACKAGE(Qt4)
ENDIF() 

IF( QT4_FOUND )
    include(${QT_USE_FILE})
ENDIF()
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#SET(QT_UI_HEADERS 
#)

SET(QT_INCLUDED_QOBJECT_HEADERS
    qtbuttonpropertybrowser.h
    qteditorfactory.h
    qtgroupboxpropertybrowser.h
    qtpropertybrowser.h
    qtpropertymanager.h
    qtvariantproperty.h
    qttreepropertybrowser.h
)

SET(QT_QOBJECT_HEADERS
    qtpropertybrowserutils_p.h
)

SET(HEADERS
    MemberProperty.hpp
    MemberPropertyManager.hpp
)

#QT_WRAP_UI(${Project} QT_PROCESSED_UI_HEADERS QT_PROCESSED_UI_SOURCE ${QT_UI_HEADERS})
QT_WRAP_CPP(${Project} QT_PROCESSED_MOC_SOURCE ${QT_QOBJECT_HEADERS})
QT_WRAP_CPP(${Project} QT_INCLUDED_PROCESSED_MOC_SOURCE ${QT_INCLUDED_QOBJECT_HEADERS})

SET(Sources
    qtbuttonpropertybrowser.cpp
    qtpropertybrowser.cpp		
    qttreepropertybrowser.cpp
    qteditorfactory.cpp		
    qtpropertybrowserutils.cpp	
    qtvariantproperty.cpp
    qtgroupboxpropertybrowser.cpp	
    qtpropertymanager.cpp
)

SET(Headers
)

# The .cpp files include the generate moc files, so, while they need to be present for the triggers,
# they shouldn't be compiled separately.
SET_SOURCE_FILES_PROPERTIES(${QT_INCLUDED_PROCESSED_MOC_SOURCE} PROPERTIES HEADER_FILE_ONLY ON)

ADD_ELVIS_STATIC_LIBRARY(${Project} 
    ${Sources} 
    ${Headers}
    ${QT_QOBJECT_HEADERS}
    #${QT_PROCESSED_UI_HEADERS}
    ${QT_INCLUDED_QOBJECT_HEADERS}
    #${QT_PROCESSED_UI_SOURCE}
    ${QT_PROCESSED_MOC_SOURCE}
    ${QT_INCLUDED_PROCESSED_MOC_SOURCE}
    ${HEADERS}
)

SET_PROPERTY(TARGET ${Project} PROPERTY FOLDER "Utilities")
INCLUDE_DIRECTORIES($<TARGET_FILE_DIR:${Project}>)

TARGET_LINK_LIBRARIES(${Project}
    ${QT_LIBRARIES}
)
