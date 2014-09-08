#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2009 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software. 
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

#==================================================
# Setup valgrind for memory checking
#==================================================

IF( NOT WIN32 )
  FIND_PROGRAM( VALGRIND_EXEC valgrind  )
ENDIF()

IF( NOT WIN32 AND VALGRIND_EXEC )

  SET( VALGRIND_FLAGS --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=50 --error-exitcode=1 --gen-suppressions=all )
  SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/ElVis.supp )
  SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/Qt.supp )
  IF( APPLE )
    SET( VALGRIND_FLAGS ${VALGRIND_FLAGS} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/OSX.supp )
  ENDIF()
    
  SET( VALGRIND_EXTRA_FLAGS CACHE STRING "Additonal valgrind flags: Usefull ones are --track-origins=yes" )
  SET( VALGRIND_COMMAND ${VALGRIND_EXEC} ${VALGRIND_FLAGS} ${VALGRIND_EXTRA_FLAGS} )

  IF( CMAKE_BUILD_TYPE )
    STRING( TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE ) 
  ENDIF()
    
  IF( BUILD_TYPE AND BUILD_TYPE MATCHES "DEBUG" )
    SET( STACKCHECK_COMMAND ${VALGRIND_EXEC} --tool=exp-sgcheck --gen-suppressions=all )
    SET( STACKCHECK_COMMAND ${STACKCHECK_COMMAND} --suppressions=${CMAKE_SOURCE_DIR}/Externals/valgrind/sgcheck.supp )
  ELSE()
    SET( STACKCHECK_COMMAND ${CMAKE_COMMAND} -E echo "Please set CMAKE_BUILD_TYPE to \\'debug\\' for stack checking of" )
  ENDIF()
    
  SET( CALLGRIND_COMMAND ${VALGRIND_EXEC} --tool=callgrind )
  
  MACRO( ADD_MEMCHECK UNIT_TEST )
    #Add the memory checking target
    ADD_CUSTOM_TARGET( ${UNIT_TEST}_memcheck COMMAND ${VALGRIND_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build> $(UNITARGS)
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    
    #Add the stack checking target
    ADD_CUSTOM_TARGET( ${UNIT_TEST}_stackcheck COMMAND ${STACKCHECK_COMMAND} $<TARGET_FILE:${UNIT_TEST}_build> $(UNITARGS)
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
  ENDMACRO()
  
  ADD_CUSTOM_TARGET( memcheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisMemCheck
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( stackcheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L ElVisStackCheck
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )

  ADD_CUSTOM_TARGET( MemAndStackCheck 
                     COMMAND ${CMAKE_CTEST_COMMAND} $(CTESTARGS) --output-on-failure --no-compress-output -L \"ElVisMemCheck|ElVisStackCheck\"
                     DEPENDS check_build
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
ELSE()
  IF( WIN32 )
    SET( NO_VALGRIND_MESSAGE "Valgrind is not available for windows." )
  ELSE()
    SET( NO_VALGRIND_MESSAGE "Please install valgrind for memory checking." )
  ENDIF()
  
  MACRO( ADD_MEMCHECK UNIT_TEST )
    ADD_CUSTOM_TARGET( ${UNIT_TEST}_memcheck COMMAND ${CMAKE_COMMAND} -E echo ${NO_VALGRIND_MESSAGE} )
    ADD_CUSTOM_TARGET( ${UNIT_TEST}_stackcheck COMMAND ${CMAKE_COMMAND} -E echo ${NO_VALGRIND_MESSAGE} )
  ENDMACRO()
  
  ADD_CUSTOM_TARGET( memcheck COMMAND ${CMAKE_COMMAND} -E echo ${NO_VALGRIND_MESSAGE} )
  ADD_CUSTOM_TARGET( stackcheck COMMAND ${CMAKE_COMMAND} -E echo ${NO_VALGRIND_MESSAGE} )
  ADD_CUSTOM_TARGET( MemAndStackCheck COMMAND ${CMAKE_COMMAND} -E echo ${NO_VALGRIND_MESSAGE} )
 
ENDIF()
