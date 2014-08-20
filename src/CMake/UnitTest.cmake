ADD_CUSTOM_TARGET( check_build )

ADD_CUSTOM_TARGET( check 
                   COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisCheck
                   DEPENDS check_build
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/XCodeFileGlob.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/Valgrind.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/Coverage.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/CallGrind.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/QTest.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/CMake/BoostTest.cmake)
