
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

//-------------------------------------------------------------------------------
//
//  whitted.cpp -- whitted's original sphere scene 
//
//-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include "random.h"
#include <iostream>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>

using namespace optix;

//-----------------------------------------------------------------------------
// 
// Whitted Scene
//
//-----------------------------------------------------------------------------

class WhittedScene : public SampleScene
{
public:
  WhittedScene() : SampleScene(), m_frame_number( 0 ), m_adaptive_aa( false ), m_width(512u), m_height(512u) {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
  Buffer getOutputBuffer();
  bool keyPressed(unsigned char key, int x, int y);

  void setAdaptiveAA( bool adaptive_aa ) { m_adaptive_aa = adaptive_aa; }

private:
  int getEntryPoint() { return m_adaptive_aa ? AdaptivePinhole: Pinhole; }
  void genRndSeeds(unsigned int width, unsigned int height);

  enum {
    Pinhole = 0,
    AdaptivePinhole = 1
  };

  void createGeometry();

  Buffer        m_rnd_seeds;
  unsigned int  m_frame_number;
  bool          m_adaptive_aa;

  unsigned int m_width;
  unsigned int m_height;
};

void WhittedScene::genRndSeeds( unsigned int width, unsigned int height )
{
  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer( seeds, width*height );
  m_rnd_seeds->unmap();
}

void WhittedScene::initScene( InitialCameraData& camera_data )
{
  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 2 );
  m_context->setStackSize( 2800 );

  m_context["max_depth"]->setInt( 10 );
  m_context["radiance_ray_type"]->setUint( 0 );
  m_context["shadow_ray_type"]->setUint( 1 );
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-4f );
  m_context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );


  m_context["output_buffer"]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height) );

  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = ptxpath( "whitted", "pinhole_camera.cu" );
  m_context->setRayGenerationProgram( Pinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     Pinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  // Adaptive Pinhole Camera ray gen and exception program
  ptx_path = ptxpath( "whitted", "adaptive_pinhole_camera.cu" );
  m_context->setRayGenerationProgram( AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );


  // Miss program
  Program miss_program = m_context->createProgramFromPTXFile( ptxpath( "whitted", "constantbg.cu" ), "miss" );
  m_context->setMissProgram( 0, miss_program );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );

  // Lights
  BasicLight lights[] = {
    { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);

  // Set up camera
  camera_data = InitialCameraData( make_float3( 8.0f, 2.0f, -4.0f ), // eye
                                   make_float3( 4.0f, 2.3f, -4.0f ), // lookat
                                   make_float3( 0.0f, 1.0f,  0.0f ), // up
                                   60.0f );                          // vfov

  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Variance buffers
  Buffer variance_sum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                       RT_FORMAT_FLOAT4,
                                                       m_width, m_height );
  memset( variance_sum_buffer->map(), 0, m_width*m_height*sizeof(float4) );
  variance_sum_buffer->unmap();
  m_context["variance_sum_buffer"]->set( variance_sum_buffer );

  Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                        RT_FORMAT_FLOAT4,
                                                        m_width, m_height );
  memset( variance_sum2_buffer->map(), 0, m_width*m_height*sizeof(float4) );
  variance_sum2_buffer->unmap();
  m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

  // Sample count buffer
  Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                                      RT_FORMAT_UNSIGNED_INT,
                                                      m_width, m_height );
  memset( num_samples_buffer->map(), 0, m_width*m_height*sizeof(unsigned int) );
  num_samples_buffer->unmap();
  m_context["num_samples_buffer"]->set( num_samples_buffer);

  // RNG seed buffer
  m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
                                               RT_FORMAT_UNSIGNED_INT,
                                               m_width, m_height );
  m_context["rnd_seeds"]->set( m_rnd_seeds );
  genRndSeeds( m_width, m_height );

  // Populate scene hierarchy
  createGeometry();

  // Prepare to run
  m_context->validate();
  m_context->compile();
}

Buffer WhittedScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

// Return whether we processed the key or not
bool WhittedScene::keyPressed(unsigned char key, int x, int y)
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

void WhittedScene::trace( const RayGenCameraData& camera_data )
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
                   static_cast<unsigned int>(buffer_height) );
}


void WhittedScene::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
  m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
  m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
  m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
  m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
  genRndSeeds( width, height );
}

void WhittedScene::createGeometry()
{
  // Create glass sphere geometry
  std::string shell_ptx( ptxpath( "whitted", "sphere_shell.cu" ) ); 
  Geometry glass_sphere = m_context->createGeometry();
  glass_sphere->setPrimitiveCount( 1u );
  glass_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( shell_ptx, "bounds" ) );
  glass_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( shell_ptx, "intersect" ) );
  glass_sphere["center"]->setFloat( 4.0f, 2.3f, -4.0f );
  glass_sphere["radius1"]->setFloat( 0.96f );
  glass_sphere["radius2"]->setFloat( 1.0f );
  
  // Metal sphere geometry
  std::string sphere_ptx( ptxpath( "whitted", "sphere.cu" ) ); 
  Geometry metal_sphere = m_context->createGeometry();
  metal_sphere->setPrimitiveCount( 1u );
  metal_sphere->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
  metal_sphere->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "robust_intersect" ) );
  metal_sphere["sphere"]->setFloat( 2.0f, 1.5f, -2.5f, 1.0f );

  // Floor geometry
  std::string pgram_ptx( ptxpath( "whitted", "parallelogram.cu" ) );
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( m_context->createProgramFromPTXFile( pgram_ptx, "bounds" ) );
  parallelogram->setIntersectionProgram( m_context->createProgramFromPTXFile( pgram_ptx, "intersect" ) );
  float3 anchor = make_float3( -16.0f, 0.01f, -8.0f );
  float3 v1 = make_float3( 32.0f, 0.0f, 0.0f );
  float3 v2 = make_float3( 0.0f, 0.0f, 16.0f );
  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  float4 plane = make_float4( normal, d );
  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  // Glass material
  Program glass_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "closest_hit_radiance" );
  Program glass_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "glass.cu" ), "any_hit_shadow" );
  Material glass_matl = m_context->createMaterial();
  glass_matl->setClosestHitProgram( 0, glass_ch );
  glass_matl->setAnyHitProgram( 1, glass_ah );

  glass_matl["importance_cutoff"]->setFloat( 1e-2f );
  glass_matl["cutoff_color"]->setFloat( 0.034f, 0.055f, 0.085f );
  glass_matl["fresnel_exponent"]->setFloat( 3.0f );
  glass_matl["fresnel_minimum"]->setFloat( 0.1f );
  glass_matl["fresnel_maximum"]->setFloat( 1.0f );
  glass_matl["refraction_index"]->setFloat( 1.4f );
  glass_matl["refraction_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["reflection_color"]->setFloat( 1.0f, 1.0f, 1.0f );
  glass_matl["refraction_maxdepth"]->setInt( 10 );
  glass_matl["reflection_maxdepth"]->setInt( 5 );
  float3 extinction = make_float3(.83f, .83f, .83f);
  glass_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
  glass_matl["shadow_attenuation"]->setFloat( 0.6f, 0.6f, 0.6f );

  // Metal material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "phong.cu" ), "any_hit_shadow" );

  Material metal_matl = m_context->createMaterial();
  metal_matl->setClosestHitProgram( 0, phong_ch );
  metal_matl->setAnyHitProgram( 1, phong_ah );
  metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
  metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
  metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
  metal_matl["phong_exp"]->setFloat( 64 );
  metal_matl["reflectivity"]->setFloat( 0.5f,  0.5f,  0.5f);

  // Checker material for floor
  Program check_ch = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "closest_hit_radiance" );
  Program check_ah = m_context->createProgramFromPTXFile( ptxpath( "whitted", "checker.cu" ), "any_hit_shadow" );
  Material floor_matl = m_context->createMaterial();
  floor_matl->setClosestHitProgram( 0, check_ch );
  floor_matl->setAnyHitProgram( 1, check_ah );

  floor_matl["Kd1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ka1"]->setFloat( 0.8f, 0.3f, 0.15f);
  floor_matl["Ks1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["Kd2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ka2"]->setFloat( 0.9f, 0.85f, 0.05f);
  floor_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["inv_checker_size"]->setFloat( 32.0f, 16.0f, 1.0f );
  floor_matl["phong_exp1"]->setFloat( 0.0f );
  floor_matl["phong_exp2"]->setFloat( 0.0f );
  floor_matl["reflectivity1"]->setFloat( 0.0f, 0.0f, 0.0f);
  floor_matl["reflectivity2"]->setFloat( 0.0f, 0.0f, 0.0f);

  // Create GIs for each piece of geometry
  std::vector<GeometryInstance> gis;
  gis.push_back( m_context->createGeometryInstance( glass_sphere, &glass_matl, &glass_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( metal_sphere,  &metal_matl,  &metal_matl+1 ) );
  gis.push_back( m_context->createGeometryInstance( parallelogram, &floor_matl,  &floor_matl+1 ) );

  // Place all in group
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( static_cast<unsigned int>(gis.size()) );
  geometrygroup->setChild( 0, gis[0] );
  geometrygroup->setChild( 1, gis[1] );
  geometrygroup->setChild( 2, gis[2] );
  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );

  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------


void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -A  | --adaptive-off                       Turn off adaptive AA\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Toggles adaptive pixel sampling on and off\n"
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  bool adaptive_aa = true;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--adaptive-off" || arg == "-A" ) {
      adaptive_aa = false;
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    WhittedScene scene;
    scene.setAdaptiveAA( adaptive_aa );
    GLUTDisplay::run( "WhittedScene", &scene, adaptive_aa ? GLUTDisplay::CDProgressive : GLUTDisplay::CDNone );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
