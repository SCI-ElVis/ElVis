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

#===============================================================================
# Do a file glob that removes XCode temporary files
#===============================================================================

#Call with for example FILE_GLOB_RECURSE( SRCS "*.h" "*.cpp" )
FUNCTION( FILE_GLOB_IMPL GLOB_TYPE RESULT )

  SET( ALL_SRCS )
  SET( i 1 ) #Skip the RESULTS argument

  WHILE( i LESS ${ARGC} )
    SET( PATTERN ${ARGV${i}} )
    FILE( ${GLOB_TYPE} SRCS ${PATTERN} )
    FILE( GLOB_RECURSE XCODE_TMP "._${PATTERN}" )

    IF (${XCODE_TMP})
      LIST( REMOVE_ITEM SRCS ${XCODE_TMP} )
    ENDIF ( )

    LIST( APPEND ALL_SRCS ${SRCS} )
    MATH( EXPR i "${i} + 1" )
  ENDWHILE()

  SET( ${RESULT} ${ALL_SRCS} PARENT_SCOPE )

ENDFUNCTION()



FUNCTION( FILE_GLOB RESULT )

  FILE_GLOB_IMPL( GLOB RESULT_TMP ${ARGN} )

  SET( ${RESULT} ${RESULT_TMP} PARENT_SCOPE )

ENDFUNCTION()

FUNCTION( FILE_GLOB_RECURSE RESULT )

  FILE_GLOB_IMPL( GLOB_RECURSE RESULT_TMP ${ARGN} )

  SET( ${RESULT} ${RESULT_TMP} PARENT_SCOPE )

ENDFUNCTION()
