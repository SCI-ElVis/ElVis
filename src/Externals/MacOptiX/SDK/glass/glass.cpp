
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
//  A glass shader example.
//
//------------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "random.h"
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <sutil.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <string.h>

using namespace optix;

//------------------------------------------------------------------------------
//
//  Glass scene 
//
//------------------------------------------------------------------------------

class GlassScene : public SampleScene
{
public:
  GlassScene( const std::string& obj_path, bool aaa, bool gg ) 
    : SampleScene(), m_obj_path( obj_path ), m_frame_number( 0u ), m_adaptive_aa( aaa ), m_green_glass( gg ) {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int depth );
  Buffer getOutputBuffer();
  bool keyPressed(unsigned char key, int x, int y);

private:
  void createContext( SampleScene::InitialCameraData& camera_data );
  void createMaterials(Material material[] );
  void createGeometry( Material material[], const std::string& path );

  // Helper functions
  void makeMaterialPrograms( Material material, const char *filename,
                                                const char *ch_program_name,
                                                const char *ah_program_name );

  int getEntryPoint() { return m_adaptive_aa ? AdaptivePinhole: Pinhole; }
  void genRndSeeds(unsigned int width, unsigned int height);

  enum {
    Pinhole = 0,
    AdaptivePinhole = 1
  };

  void createGeometry();

  Buffer        m_rnd_seeds;
  std::string   m_obj_path;
  unsigned int  m_frame_number;
  bool          m_adaptive_aa;
  bool          m_green_glass;

  static unsigned int WIDTH;
  static unsigned int HEIGHT;
};

unsigned int GlassScene::WIDTH  = 512u;
unsigned int GlassScene::HEIGHT = 384u;


void GlassScene::genRndSeeds( unsigned int width, unsigned int height )
{
  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer(seeds, width*height);
  m_rnd_seeds->unmap();
}

void GlassScene::initScene( InitialCameraData& camera_data ) 
{
  try {
    optix::Material material[2];
    createContext( camera_data );
    createMaterials( material );
    createGeometry( material, m_obj_path );

    m_context->validate();
    m_context->compile();

  } catch( Exception& e ) {
    sutilReportError( e.getErrorString().c_str() );
    exit( 2 );
  }
}


Buffer GlassScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void GlassScene::trace( const RayGenCameraData& camera_data )
{
  if ( m_camera_changed ) {
    m_frame_number = 0u;
    m_camera_changed = false;
  }

  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
  m_context["frame_number"]->setUint( m_frame_number++ );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( getEntryPoint(),
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height)
                   );
}


void GlassScene::doResize( unsigned int width, unsigned int height )
{
  // We need to update buffer sizes if resized (output_buffer handled in base class)
  m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
  m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
  m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
  m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
  genRndSeeds( width, height );
}


// Return whether we processed the key or not
bool GlassScene::keyPressed(unsigned char key, int x, int y)
{
  switch (key)
  {
  case 'a':
    m_adaptive_aa = !m_adaptive_aa;
    m_camera_changed = true;
    GLUTDisplay::setContinuousMode( m_adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDNone );
    return true;
  }
  return false;
}


void  GlassScene::createContext( InitialCameraData& camera_data )
{
  // Context
  m_context->setEntryPointCount( 2 );
  m_context->setRayTypeCount( 2 );
  m_context->setStackSize( 2400 );

  m_context["scene_epsilon"]->setFloat( 1.e-3f );
  m_context["radiance_ray_type"]->setUint( 0u );
  m_context["shadow_ray_type"]->setUint( 1u );
  m_context["max_depth"]->setInt( 10 );
  m_context["frame_number"]->setUint( 0u );

  // Output buffer.
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT );
  output_buffer->set(buffer);
  
  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = ptxpath( "glass", "pinhole_camera.cu" );
  m_context->setRayGenerationProgram( Pinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     Pinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  // Adaptive Pinhole Camera ray gen and exception program
  ptx_path = ptxpath( "glass", "adaptive_pinhole_camera.cu" );
  m_context->setRayGenerationProgram( AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );


  // Used by both exception programs
  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 1.0f );

  // Miss program.
  ptx_path = ptxpath( "glass", "gradientbg.cu" );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
  m_context["background_light"]->setFloat( 1.0f, 1.0f, 1.0f );
  m_context["background_dark"]->setFloat( 0.3f, 0.3f, 0.3f );

  // align background's up direction with camera's look direction
  float3 bg_up = make_float3(-14.0f, -14.0f, -7.0f);
  bg_up = normalize(bg_up);

  // tilt the background's up direction in the direction of the camera's up direction
  bg_up.y += 1.0f;
  bg_up = normalize(bg_up);
  m_context["up"]->setFloat( bg_up.x, bg_up.y, bg_up.z );
  
  // Set up camera
  camera_data = InitialCameraData( make_float3( 14.0f, 14.0f, 14.0f ), // eye
                                   make_float3( 0.0f, 7.0f, 0.0f ),    // lookat
                                   make_float3( 0.0f, 1.0f, 0.0f ),    // up
                                   45.0f );                            // vfov
  
  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Variance buffers
  Buffer variance_sum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                       RT_FORMAT_FLOAT4,
                                                       WIDTH, HEIGHT );
  memset( variance_sum_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
  variance_sum_buffer->unmap();
  m_context["variance_sum_buffer"]->set( variance_sum_buffer );

  Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                        RT_FORMAT_FLOAT4,
                                                        WIDTH, HEIGHT );
  memset( variance_sum2_buffer->map(), 0, WIDTH*HEIGHT*sizeof(float4) );
  variance_sum2_buffer->unmap();
  m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

  // Sample count buffer
  Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                      RT_FORMAT_UNSIGNED_INT,
                                                      WIDTH, HEIGHT );
  memset( num_samples_buffer->map(), 0, WIDTH*HEIGHT*sizeof(unsigned int) );
  num_samples_buffer->unmap();
  m_context["num_samples_buffer"]->set( num_samples_buffer);

  // RNG seed buffer
  m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                       RT_FORMAT_UNSIGNED_INT,
                                       WIDTH, HEIGHT );
  genRndSeeds( WIDTH, HEIGHT );
  m_context["rnd_seeds"]->set( m_rnd_seeds );
}


void GlassScene::createMaterials( Material material[] )
{
  material[0] = m_context->createMaterial();
  material[1] = m_context->createMaterial();

  makeMaterialPrograms( material[0], "glass.cu", "closest_hit_radiance", "any_hit_shadow");

  material[0]["importance_cutoff"  ]->setFloat( 0.01f );
  material[0]["cutoff_color"       ]->setFloat( 0.2f, 0.2f, 0.2f );
  material[0]["fresnel_exponent"   ]->setFloat( 4.0f );
  material[0]["fresnel_minimum"    ]->setFloat( 0.1f );
  material[0]["fresnel_maximum"    ]->setFloat( 1.0f );
  material[0]["refraction_index"   ]->setFloat( 1.4f );
  material[0]["refraction_color"   ]->setFloat( 0.99f, 0.99f, 0.99f );
  material[0]["reflection_color"   ]->setFloat( 0.99f, 0.99f, 0.99f );
  material[0]["refraction_maxdepth"]->setInt( 10 );
  material[0]["reflection_maxdepth"]->setInt( 5 );
  float3 extinction = m_green_glass ? make_float3(.80f, .89f, .75f) : make_float3(1);
  material[0]["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
  material[0]["shadow_attenuation"]->setFloat( 1.0f, 1.0f, 1.0f );

  // Checkerboard to aid positioning, not used in final setup.
  makeMaterialPrograms( material[1], "checkerboard.cu", "closest_hit_radiance", "any_hit_shadow");
  
  material[1]["tile_size"       ]->setFloat( 1.0f, 1.0f, 1.0f );
  material[1]["tile_color_dark" ]->setFloat( 1.0f, 0.0f, 0.0f );
  material[1]["tile_color_light"]->setFloat( 1.0f, 1.0f, 0.0f );
  material[1]["light_direction" ]->setFloat( 0.0f, 1.0f, 1.0f );
}


void GlassScene::createGeometry( Material material[], const std::string& path )
{
  // Load OBJ files and set as geometry groups
  GeometryGroup geomgroup[3] = { m_context->createGeometryGroup(),
                                 m_context->createGeometryGroup(),
                                 m_context->createGeometryGroup() };

  // Set transformations
  const float matrix_0[4*4] = { 1,  0,  0,  0, 
                                0,  1,  0,  0, 
                                0,  0,  1, -5, 
                                0,  0,  0,  1 };

  const float matrix_1[4*4] = { 1,  0,  0,  0, 
                                0,  1,  0,  0, 
                                0,  0,  1,  0, 
                                0,  0,  0,  1 };

  const float matrix_2[4*4] = { 1,  0,  0, -5, 
                                0,  1,  0,  0, 
                                0,  0,  1,  0, 
                                0,  0,  0,  1 };

  const optix::Matrix4x4 m0( matrix_0 );
  const optix::Matrix4x4 m1( matrix_1 );
  const optix::Matrix4x4 m2( matrix_2 );

  std::string prog_path = std::string(sutilSamplesPtxDir()) + "/glass_generated_triangle_mesh_iterative.cu.ptx";
  Program mesh_intersect = m_context->createProgramFromPTXFile( prog_path, "mesh_intersect" );

  ObjLoader objloader0( (path + "/cognacglass.obj").c_str(), m_context, geomgroup[0], material[0] );
  objloader0.setIntersectProgram( mesh_intersect );
  objloader0.load( m0 );
  ObjLoader objloader1( (path + "/wineglass.obj").c_str(),   m_context, geomgroup[1], material[0] );
  objloader1.setIntersectProgram( mesh_intersect );
  objloader1.load( m1 );
  ObjLoader objloader2( (path + "/waterglass.obj").c_str(),  m_context, geomgroup[2], material[0] );
  objloader2.setIntersectProgram( mesh_intersect );
  objloader2.load( m2 );
  
  GeometryGroup maingroup = m_context->createGeometryGroup();
  maingroup->setChildCount( 3 );
  maingroup->setChild( 0, geomgroup[0]->getChild(0) );
  maingroup->setChild( 1, geomgroup[1]->getChild(0) );
  maingroup->setChild( 2, geomgroup[2]->getChild(0) );
  maingroup->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );
  m_context["top_object"]->set( maingroup );
}


void GlassScene::makeMaterialPrograms( Material material, const char *filename, 
                                                          const char *ch_program_name,
                                                          const char *ah_program_name )
{
  Program ch_program = m_context->createProgramFromPTXFile( ptxpath("glass", filename), ch_program_name );
  Program ah_program = m_context->createProgramFromPTXFile( ptxpath("glass", filename), ah_program_name );

  material->setClosestHitProgram( 0, ch_program );
  material->setAnyHitProgram( 1, ah_program );
}


//------------------------------------------------------------------------------
//
//  main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"

    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj-path <path>                    Specify path to OBJ files\n"
    << "  -A  | --adaptive-off                       Turn off adaptive AA\n"
    << "  -g  | --green                              Make the glass green\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Toggles adaptive pixel sampling on and off\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main(int argc, char* argv[])
{
  GLUTDisplay::init( argc, argv );

  bool adaptive_aa = true;  // Default to true for now
  bool green_glass = false;
  std::string obj_path;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--adaptive-off" || arg == "-A" ) {
      adaptive_aa = false;
    } else if ( arg == "--green" || arg == "-g" ) {
      green_glass = true;
    } else if ( arg == "--obj-path" || arg == "-o" ) {
      if ( i == argc-1 ) {
        printUsageAndExit( argv[0] );
      }
      obj_path = argv[++i];
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( obj_path.empty() ) {
    obj_path = std::string( sutilSamplesDir() ) + "/glass";
  }

  try {
    GlassScene scene( obj_path, adaptive_aa, green_glass );
    GLUTDisplay::setTextColor( make_float3( 0.2f ) );
    GLUTDisplay::setTextShadowColor( make_float3( 0.9f ) );
    GLUTDisplay::run( "GlassScene", &scene, adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDNone );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
