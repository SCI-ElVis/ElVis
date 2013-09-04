
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <ObjLoader.h>
#include <PlyLoader.h>
#include <iostream>


////////////////////////////////////////////
// 
//  Entry point from sutil api 
//
////////////////////////////////////////////

extern "C"
RTresult loadModel( const char* fname, RTcontext context, RTmaterial mat, RTgeometrygroup* geometrygroup )
{

  // Ensure we have a valid geometry group 
  if ( !(*geometrygroup) ) {
    RTresult result = rtGeometryGroupCreate( context, geometrygroup );
    if ( result != RT_SUCCESS ) {
      std::cerr << "loadModel(): rtGeometryGroupCreate failed. " << std::endl;
      return result;
    }
  } else {
    unsigned int num_children;
    RTresult result = rtGeometryGroupGetChildCount( *geometrygroup, &num_children );
    if ( result != RT_SUCCESS ) {
      std::cerr << "loadModel(): rtGeometryGroupGetChildCount failed. " << std::endl;
      return result;
    }
    if ( num_children != 0 ) {
      std::cerr << "loadModel() - geometry group has preexisting children" << std::endl; 
      return RT_ERROR_INVALID_VALUE;
    }
  }

  // Load the file
  if ( ObjLoader::isMyFile( fname ) ) {
    try {
      if ( mat )  {
        ObjLoader loader( fname,
                          optix::Context::take( context ),
                          optix::GeometryGroup::take( *geometrygroup ),
                          optix::Material::take( mat ) );
        loader.load();
      } else {
        ObjLoader loader( fname,
                          optix::Context::take( context ),
                          optix::GeometryGroup::take( *geometrygroup ) );
        loader.load();
      }
    } catch( optix::Exception& e ) {
      std::cerr << " loadModel() failed: '" << e.getErrorString() << "'" 
                << std::endl;
      return RT_ERROR_UNKNOWN;
    } catch( ... ) {
      std::cerr << " loadModel() failed: error unknown" << std::endl; 
      return RT_ERROR_UNKNOWN;
    }
    return RT_SUCCESS;

  } else if( PlyLoader::isMyFile( fname ) ) { 
    try {
      PlyLoader loader( fname,
                        optix::Context::take( context ),
                        optix::GeometryGroup::take( *geometrygroup ),
                        optix::Material::take( mat ) );
      loader.load();
    } catch( optix::Exception& e ) {
      std::cerr << " loadModel() failed: '" << e.getErrorString() << "'" 
                << std::endl;
      return RT_ERROR_UNKNOWN;
    } catch( ... ) {
      std::cerr << " loadModel() failed: error unknown" << std::endl; 
      return RT_ERROR_UNKNOWN;
    }
    return RT_SUCCESS;

  } else {
    std::cerr << "loadModel: '" << fname << "' extension not recognized." 
              << std::endl;
    return RT_ERROR_INVALID_VALUE;
  }
}
