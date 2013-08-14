
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

//------------------------------------------------------------------------------
//
//  simpleAnimation.cpp -- Renders an OBJ model with a time-varying warping
//                         transform.
//
//------------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <sutil.h>
#include "commonStructs.h"
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <string>
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace optix;


//------------------------------------------------------------------------------
//
// SimpleAnimationScene definition
//
//------------------------------------------------------------------------------

class SimpleAnimationScene : public SampleScene
{
public:
  SimpleAnimationScene( const std::string& filename )
    : m_filename( filename ), m_model_scale( 1.0f )
  { m_vertices = 0; }

  ~SimpleAnimationScene()
  { delete[] m_vertices; }

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();

private:
  void initAnimation( );
  void updateGeometry( );

  std::string   m_filename;
  GeometryGroup m_geometry_group;
  float3*       m_vertices;
  float         m_model_scale;

  const static int WIDTH;
  const static int HEIGHT;
};

const int SimpleAnimationScene::WIDTH  = 1024;
const int SimpleAnimationScene::HEIGHT = 1024;


void SimpleAnimationScene::initScene( InitialCameraData& camera_data )
{
  // Setup context
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 720 );

  m_context[ "radiance_ray_type"]->setUint( 0u );
  m_context[ "shadow_ray_type" ]->setUint( 1u );
  m_context[ "scene_epsilon" ]->setFloat( 1.e-4f );
  m_context[ "max_depth" ]->setInt( 5 );
  m_context[ "ambient_light_color" ]->setFloat( 0.4f, 0.4f, 0.4f );

  // Output buffer
  m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

  // Lights buffer
  BasicLight lights[] = {
    { make_float3( 60.0f, 60.0f, 60.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();
  m_context[ "lights" ]->set( light_buffer );

  // Ray generation program
  std::string ptx_path = ptxpath( "simpleAnimation", "pinhole_camera.cu" );
  Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
  m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "simpleAnimation", "constantbg.cu" ), "miss" ) );
  m_context[ "bg_color" ]->setFloat(  0.1f, 0.7f, 0.7f );
  
  // Load OBJ scene 
  m_geometry_group = m_context->createGeometryGroup();
  ObjLoader loader( m_filename.c_str(), m_context, m_geometry_group );
  loader.load();

  // Select a faster AS builder than the ObjLoader default.
  Acceleration accel = m_geometry_group->getAcceleration();
  accel->setBuilder( "Lbvh" );

  initAnimation( );

  m_context[ "top_object" ]->set( m_geometry_group ); 
  m_context[ "top_shadower" ]->set( m_geometry_group ); 

  // Set up camera
  optix::Aabb aabb = loader.getSceneBBox();
  m_model_scale = fmaxf( aabb.extent( 0 ), aabb.extent( 1 ) ); // max of x,y components
  float3 eye = aabb.center();
  eye.z += 2.0f * m_model_scale + 0.05f;
  camera_data = InitialCameraData( eye,                             // eye
                                   aabb.center(),                   // lookat
                                   make_float3( 0.0f, 1.0f, 0.0f ), // up
                                   45.0f );                         // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Prepare to run 
  m_context->validate();
  m_context->compile();
}


void SimpleAnimationScene::initAnimation( )
{
  //We know that we have (only) one group in this example
  GeometryInstance geometryInstance = m_geometry_group->getChild( 0 );
  Geometry geometry = geometryInstance->getGeometry();

  /*
    All that we want to do here is to copy
    the original vertex positions we get from
    the objLoader to an array. We use this
    array to always have access to the 
    original vertex position; the values in the
    objLoader buffer will be altered per frame.
  */

  //Query vertex buffer
  Buffer vertexBuffer = geometry["vertex_buffer"]->getBuffer();
  
  //Query number of vertices in the buffer
  RTsize numVertices;
  vertexBuffer->getSize( numVertices );

  //Get a pointer to the buffer data
  float3* original_vertices = (float3*)vertexBuffer->map();

  //Allocate our storage array and copy values
  m_vertices = new float3[numVertices];
  memcpy(m_vertices, original_vertices, numVertices * sizeof(float3));

  //Unmap buffer
  vertexBuffer->unmap();
}


void SimpleAnimationScene::updateGeometry( )
{
  //We know that we have (only) one group in this example
  GeometryInstance geometryInstance = m_geometry_group->getChild( 0 );
  Geometry geometry = geometryInstance->getGeometry();

  /* 
      All we want to do here is to add a simple sin(x) offset
      to the vertices y-position.
  */

  Buffer vertexBuffer = geometry["vertex_buffer"]->getBuffer();
  float3* new_vertices = (float3*)vertexBuffer->map();

  RTsize numVertices;
  vertexBuffer->getSize( numVertices );

  static float t = 0.0f;

  //We don't have to set x and z here in this example
  for(unsigned int v = 0; v < numVertices; v++)
  {
    new_vertices[v].y = m_vertices[v].y + ( sinf( m_vertices[v].x / m_model_scale * 3.0f + t ) * m_model_scale * 0.7f );
  }

  t += 0.1f;

  vertexBuffer->unmap();

  /*
    Vertices are changed now; we have to tell the
    corresponding acceleration structure that it 
    has to be rebuild.

    Mark the accel structure and geometry as dirty.
  */
  geometry->markDirty();
  m_geometry_group->getAcceleration()->markDirty();
}


void SimpleAnimationScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  //Update the vertex positions before rendering
  updateGeometry( );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width),
                      static_cast<unsigned int>(buffer_height) );
}

Buffer SimpleAnimationScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


//------------------------------------------------------------------------------
//
//  main driver
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << std::endl;
  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}

int main( int argc, char** argv ) 
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  std::string filename;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "--obj" || arg == "-o" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      filename = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }
  if( filename.empty() ) {
    filename = std::string( sutilSamplesDir() ) + "/simpleAnimation/cow.obj";
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    SimpleAnimationScene scene( filename );
    GLUTDisplay::run( "SimpleAnimationScene", &scene, GLUTDisplay::CDAnimated );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
