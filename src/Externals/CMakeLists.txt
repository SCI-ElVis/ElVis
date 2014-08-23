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

MESSAGE(STATUS "Configuring External Packages")

##################################################
# ADDING ZLIB
##################################################

MESSAGE(STATUS "Configuring zlib")
ADD_SUBDIRECTORY(zlib)

##################################################
# ADDING LIBPNG
##################################################

MESSAGE(STATUS "Configuring libpng")
ADD_SUBDIRECTORY(libpng)

##################################################
# ADDING GLEW
##################################################

MESSAGE(STATUS "Configuring glew")
ADD_SUBDIRECTORY(glew)

##################################################
# ADDING TINYXML
##################################################

MESSAGE(STATUS "Configuring tinyxml")
ADD_SUBDIRECTORY(tinyxml)

##################################################
# Adding BOOST
##################################################

MESSAGE(STATUS "Configuring boost")
FIND_PACKAGE( Boost COMPONENTS unit_test_framework system filesystem serialization thread program_options )
IF( Boost_FOUND )
  SET(BOOST_LIBRARY ${Boost_LIBRARIES} CACHE INTERNAL "Boost library name." FORCE)
ELSE()
  MESSAGE(STATUS "Missing at least one of: unit_test_framework system filesystem serialization thread program_options" )
  MESSAGE(STATUS "Compiling boost from src/Externals/boost" )

  UNSET( Boost_DIR CACHE )
  UNSET( Boost_INCLUDE_DIR CACHE )
  UNSET( Boost_LIBRARY_DIR CACHE )
 
  UNSET( Boost_SYSTEM_LIBRARY_DEBUG CACHE )
  UNSET( Boost_SYSTEM_LIBRARY_RELEASE CACHE )
  
  UNSET( Boost_FILESYSTEM_LIBRARY_DEBUG CACHE )
  UNSET( Boost_FILESYSTEM_LIBRARY_RELEASE CACHE )

  UNSET( Boost_SERIALIZATION_LIBRARY_DEBUG CACHE )
  UNSET( Boost_SERIALIZATION_LIBRARY_RELEASE CACHE )
  
  UNSET( Boost_PROGRAM_OPTIONS_LIBRARY_DEBUG CACHE )
  UNSET( Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE CACHE )

  UNSET( Boost_THREAD_LIBRARY_DEBUG CACHE )
  UNSET( Boost_THREAD_LIBRARY_RELEASE CACHE )
  
  UNSET( Boost_UNIT_TEST_FRAMEWORK_LIBRARY_DEBUG CACHE )
  UNSET( Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE CACHE )

  ADD_SUBDIRECTORY(boost)
ENDIF()

##################################################
# ADDING OptiX
##################################################

#MESSAGE(STATUS "Configuring OptiX")
#ADD_SUBDIRECTORY(OptiX)


MESSAGE(STATUS "Configuring ElVis")
