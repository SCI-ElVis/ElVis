
/*
 * Copyright (c) 2010 NVIDIA Corporation.  All rights reserved.
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

//--------------------------------------------------------------------------------------------------
//
//  simpleGLTexInterop.cpp: Renders an Obj model textured with a simple OpenGL checkerboard texture.
//
//--------------------------------------------------------------------------------------------------

// Models and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.
// http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html

#ifdef __APPLE__
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  include <GL/gl.h>
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <sutil.h>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "MeshScene.h"

using namespace optix;

enum AccelMode {
  AM_SBVH=0,
  AM_BVH,
  AM_KD
};

class ObjScene : public MeshScene
{
public:
  ObjScene( const std::string& filename, bool accel_caching_on, AccelMode accel_mode ):
      MeshScene( false, accel_caching_on, false ),
      m_accel_mode( accel_mode ),
      m_frame( 0 )
  {
    setMesh( filename.c_str() );
  }

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual void   cleanUp();
  virtual Buffer getOutputBuffer();

private:
  AccelMode       m_accel_mode;
  float           m_scene_epsilon;
  TextureSampler  m_sampler;
  int             m_frame;

  const static int TEX_WIDTH;
  const static int TEX_HEIGHT;
};

const int ObjScene::TEX_WIDTH = 256;
const int ObjScene::TEX_HEIGHT = 256;

void ObjScene::initScene( InitialCameraData& camera_data )
{
  // Setup context
  m_context->setRayTypeCount( 1 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 400 );

  m_context[ "radiance_ray_type"]->setUint( 0u );

  GLuint texId = 0;
  glGenTextures( 1, &texId );
  glBindTexture( GL_TEXTURE_2D, texId );

  GLfloat img[TEX_HEIGHT][TEX_WIDTH][4];

  //Create a simple checkerboard texture (from OpenGL Programming Guide)
  for( int j = 0; j < TEX_HEIGHT; j++ ) {
    for( int i = 0; i < TEX_WIDTH; i++ ) {
      GLfloat c = ( ( ( i & 0x8 ) == 0 ) ^ ( ( j & 0x8 ) == 0 ) ) * 1.0f;
      img[ i ][ j ][ 0 ] = 1.0f;
      img[ i ][ j ][ 1 ] = c;
      img[ i ][ j ][ 2 ] = c;
      img[ i ][ j ][ 3 ] = 1.0f;
    }
  }

  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, TEX_WIDTH, TEX_HEIGHT, 0, GL_RGBA, GL_FLOAT, img );
  glBindTexture( GL_TEXTURE_2D, 0 );

  if( glGetError( ) != 0 )
    throw;

  // Create a texture sampler with the OpenGL texture as input.
  m_sampler = m_context->createTextureSamplerFromGLImage( texId, RT_TARGET_GL_TEXTURE_2D );
  m_sampler->setWrapMode( 0, RT_WRAP_REPEAT );
  m_sampler->setWrapMode( 1, RT_WRAP_REPEAT );
  m_sampler->setWrapMode( 2, RT_WRAP_REPEAT );
  m_sampler->setIndexingMode( RT_TEXTURE_INDEX_NORMALIZED_COORDINATES );
  m_sampler->setReadMode( RT_TEXTURE_READ_NORMALIZED_FLOAT );
  m_sampler->setMaxAnisotropy( 1.0f );
  m_sampler->setFilteringModes( RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE );

  m_context["tex"]->setTextureSampler( m_sampler );
  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT) );

  // Ray generation program
  const std::string ptx_path = ptxpath( "simpleGLTexInterop", "pinhole_camera.cu" );
  const std::string camera_name = "pinhole_camera"; 

  Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "simpleGLTexInterop", "constantbg.cu" ), "miss" ) );
  m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f );

  // Load OBJ scene 
  m_geometry_group = m_context->createGeometryGroup();

  ObjLoader* loader = 0;

  Material mat = m_context->createMaterial();
  mat->setClosestHitProgram( 0, m_context->createProgramFromPTXFile(ptxpath("simpleGLTexInterop", "simple_tex_shader.cu"),"closest_hit_radiance") );
  loader = new ObjLoader( m_filename.c_str(), m_context, m_geometry_group, mat );
  loader->load();
 
  m_context[ "top_object" ]->set( m_geometry_group );
  m_context[ "top_shadower" ]->set( m_geometry_group );

  // Load acceleration structure from a file if that was enabled on the
  // command line, and if we can find a cache file. Note that the type of
  // acceleration used will be overridden by what is found in the file.
  loadAccelCache();

  // Set up camera
  optix::Aabb aabb = loader->getSceneBBox();
  float max_dim = fmaxf( aabb.extent( 0 ), aabb.extent( 1 ) ); // max of x,y components
  float3 eye = aabb.center();
  eye.y += 2.0f * max_dim;

  camera_data = InitialCameraData( eye,                             // eye
                                   aabb.center(),                   // lookat
                                   make_float3( 0.0f, 0.0f, 1.0f ), // up
                                   30.0f );                         // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  m_scene_epsilon = 1.e-4f * max_dim;
  m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );

  // Prepare to run 
  m_context->validate();
  m_context->compile();
  
  std::cout << "\nModels and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.\n";
  std::cout << "http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html\n\n";

  // Clean up.
  delete loader;
}


void ObjScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );
}


void ObjScene::cleanUp()
{
  // Store the acceleration cache if required.
  saveAccelCache();
  SampleScene::cleanUp();
}


Buffer ObjScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -c  | --cache                              Turn on acceleration structure caching\n"
    << "        --bvh                                Use BVH acceleration instead of SBVH\n"
    << "        --kd                                 Use kd-tree acceleration instead of SBVH\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}

int main( int argc, char** argv ) 
{
  GLUTDisplay::init( argc, argv );
  
  bool        accel_caching_on = false;
  std::string filename         = "";
  AccelMode   accel_mode       = AM_SBVH;

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "-c" || arg == "--cache" ) {
      accel_caching_on = true;
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] ); 
    } else if( arg == "--bvh" ) {
      accel_mode = AM_BVH;
    } else if( arg == "--kd" ) {
      accel_mode = AM_KD;
    } else {
      std::cerr << "Unknown option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }
  if ( filename.empty() ) {
    filename = std::string( sutilSamplesDir() ) + "/simpleGLTexInterop/Koi.obj";
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    ObjScene scene( filename, accel_caching_on, accel_mode );
    GLUTDisplay::run( "simpleGLTexInterop", &scene );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
