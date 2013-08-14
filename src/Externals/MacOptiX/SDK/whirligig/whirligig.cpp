
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
//  whirligig.cpp -- crazy sphere scene 
//
//-------------------------------------------------------------------------------

#include "ringsOfSpheres.h"
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "random.h"
#include "helpers.h"
#include <iostream>
#include <GLUTDisplay.h>
#include "commonStructs.h"
#include <SunSky.h>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace optix;

inline float FRand() { return float( rand() ) / ( float( RAND_MAX ) ); } // A random number on 0.0 to 1.0.

//-----------------------------------------------------------------------------
// 
// Whirligig Scene
//
//-----------------------------------------------------------------------------

const size_t maxSpheres = 400u;

// Return 1 to ask for abort. 0 to continue.
// An RTtimeoutcallback.
int timeoutCallback()
{
  // int answer = rand() & 0x1;
  int answer = 0;
  // std::cerr << "Called back: " << answer << "\n";

  return answer;
}

class WhirligigScene : public SampleScene
{
public:
  WhirligigScene( GLUTDisplay::contDraw_E continuous_mode ) 
  : SampleScene()
  , m_frame_number( 0 )
  , m_width( 512u )
  , m_height( 512u )
  , m_all_rings( Wave_t( 0.5f, 1.0f, 0.0f, 0.8f ), Wave_t( 1.0f, 0.15f, 7.8f, 0.0f ), 0.1f, 7, 100, 8.0f ) // hardwired for benchmarking
  , m_continuous_mode( continuous_mode )
  , m_sun_scale( 1.0f / 1600000.0f )
  , m_sky_scale( 4.0f / 100.0f )
  , m_time_of_last_change( 0.0 )
  , m_allow_random_changes( true )

  {
    // Set up how the rings spin
    // hardwired for benchmarking
    size_t r = m_all_rings.Rings.size() / 2u;
    m_all_rings.Rings[r].ringRot.useWave = true;
    m_all_rings.Rings[r].ringRot.rotAccelWave = Wave_t( 0.005f, 0.01f, 1.5f );

    if( m_all_rings.Rings.size() > 5 ) {
      r = m_all_rings.Rings.size() - 1u;
      m_all_rings.Rings[r].ringRot.useWave = true;
      m_all_rings.Rings[r].ringRot.rotAccelWave = Wave_t( -0.004f, 0.0213f, 0.2f );
    }
  }

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }
  Buffer getOutputBuffer();
  bool   keyPressed( unsigned char key, int x, int y );

  void   updateScene();

private:
  int    getEntryPoint() { return m_continuous_mode == GLUTDisplay::CDProgressive ? AdaptivePinhole : Pinhole; }
  void   genRndSeeds(unsigned int width, unsigned int height);

  enum {
    Pinhole = 0,
    AdaptivePinhole = 1
  };

  void createGeometry();
  void fillInSpheres();
  void makeRings();
  void randRingVel();

  void updateLights();

  Buffer       m_rnd_seeds;
  Buffer       m_light_buffer;
  unsigned int m_frame_number;

  unsigned int m_width;
  unsigned int m_height;

  Acceleration m_sphere_accel;
  Buffer       m_sphere_buffer;
  Buffer       m_sphere_mat_buffer;
  Geometry     m_sphereG;
  std::vector<Material> m_materials;

  allRings_t   m_all_rings;
  GLUTDisplay::contDraw_E m_continuous_mode;

  PreethamSunSky m_sun_sky;
  double         m_sun_scale;
  double         m_sky_scale;

  double         m_time_of_last_change;
  bool           m_allow_random_changes;

};

void WhirligigScene::genRndSeeds( unsigned int width, unsigned int height )
{
  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer(seeds, width*height);
  m_rnd_seeds->unmap();
}

void WhirligigScene::updateLights()
{
  // Use the sky model up direction color as the ambient color
  float3 sky_color1 = m_sun_sky.skyColor( m_sun_sky.getUpDir(),  false )       * (float)m_sky_scale;
  float3 sky_color2 = m_sun_sky.skyColor( m_sun_sky.getSunDir()*0.99f, false ) * (float)m_sky_scale;
  m_context["ambient_light_color"]->setFloat( (sky_color1 + sky_color2) * 0.5f );

  // Lights
  BasicLight sunBasicLight;
  sunBasicLight.pos   = m_sun_sky.getSunDir() * 10000.0f;
  sunBasicLight.color = m_sun_sky.sunColor() * (float)m_sun_scale;
  sunBasicLight.casts_shadow = 1;

  memcpy( m_light_buffer->map(), &sunBasicLight, sizeof( sunBasicLight ) );
  m_light_buffer->unmap();
}


void WhirligigScene::initScene( InitialCameraData& camera_data )
{
  // context 
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 2 );
  m_context->setStackSize( 1920 );

  rtContextSetTimeoutCallback( m_context->get(), timeoutCallback, 0.1 );

  m_context["max_depth"]->setInt( 10 );
  m_context["radiance_ray_type"]->setUint( 0 );
  m_context["shadow_ray_type"]->setUint( 1 );
  m_context["frame_number"]->setUint( 0u );
  m_context["scene_epsilon"]->setFloat( 1.e-4f );

  m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height ) );

  // Pinhole Camera ray gen and exception program
  std::string         ptx_path = ptxpath( "whirligig", "pinhole_camera.cu" );
  m_context->setRayGenerationProgram( Pinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     Pinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  // Adaptive Pinhole Camera ray gen and exception program
  ptx_path = ptxpath( "whirligig", "adaptive_pinhole_camera.cu" );
  m_context->setRayGenerationProgram( AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" ) );
  m_context->setExceptionProgram(     AdaptivePinhole, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );

  // Miss program
  ptx_path = ptxpath( "whirligig", "sunsky.cu" );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
  m_context["sky_scale"]->setFloat( make_float3( (float)m_sky_scale ) );
  m_context["sky_up"   ]->setFloat( make_float3( 0.0f, 1.0f, 0.0f ) );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

  // Lights
  m_light_buffer = m_context->createBuffer( RT_BUFFER_INPUT );
  m_light_buffer->setFormat( RT_FORMAT_USER );
  m_light_buffer->setElementSize( sizeof( BasicLight ) );
  m_light_buffer->setSize( 1u );
  m_context["lights"]->set( m_light_buffer );

  m_sun_sky.setSunTheta( 0.9f );
  m_sun_sky.setSunPhi( 2.85f );
  m_sun_sky.setTurbidity( 2.5f );
  m_sun_sky.setVariables( m_context );
  updateLights();

  // Set up camera
  camera_data = InitialCameraData( make_float3( -3.0f, 18.0f,  2.0f ), // eye
    make_float3( 0.0f, 0.0f,  0.0f ), // lookat
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
  memset( variance_sum_buffer->map(), 0, m_width*m_height*sizeof( float4 ) );
  variance_sum_buffer->unmap();
  m_context["variance_sum_buffer"]->set( variance_sum_buffer );

  Buffer variance_sum2_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
    RT_FORMAT_FLOAT4,
    m_width, m_height );
  memset( variance_sum2_buffer->map(), 0, m_width*m_height*sizeof( float4 ) );
  variance_sum2_buffer->unmap();
  m_context["variance_sum2_buffer"]->set( variance_sum2_buffer );

  // Sample count buffer
  Buffer num_samples_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
    RT_FORMAT_UNSIGNED_INT,
    m_width, m_height );
  memset( num_samples_buffer->map(), 0, m_width*m_height*sizeof( unsigned int ) );
  num_samples_buffer->unmap();
  m_context["num_samples_buffer"]->set( num_samples_buffer );

  // RNG seed buffer
  m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
    RT_FORMAT_UNSIGNED_INT,
    m_width, m_height );
  m_context["rnd_seeds"]->set( m_rnd_seeds );
  genRndSeeds( m_width, m_height );

  // With Unix rand(), a small magnitude change in input seed yields a small change in the first random number. Duh!
  unsigned int tim = static_cast<unsigned int>( time( 0 ) );
  unsigned int tim2 = ( ( tim & 0xff ) << 24 ) | ( ( tim & 0xff00 ) << 8 ) | ( ( tim & 0xff0000 ) >> 8 ) | ( ( tim & 0xff000000 ) >> 24 );
  srand( tim2 );

  // Populate scene hierarchy
  createGeometry();

  // Prepare to run
  m_context->validate();
  m_context->compile();
}


Buffer WhirligigScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

// Return whether we processed the key or not
bool WhirligigScene::keyPressed( unsigned char key, int x, int y )
{
  unsigned int tmp_fn = m_frame_number;
  m_frame_number = 0;

  // Restart the progressive timer
  GLUTDisplay::setContinuousMode( m_continuous_mode );

  switch ( key )
  {
  case 'a': {
    if( m_continuous_mode == GLUTDisplay::CDNone ) m_continuous_mode = GLUTDisplay::CDAnimated;
    else if( m_continuous_mode == GLUTDisplay::CDProgressive ) m_continuous_mode = GLUTDisplay::CDNone;
    else if( m_continuous_mode == GLUTDisplay::CDAnimated ) m_continuous_mode = GLUTDisplay::CDProgressive;

    GLUTDisplay::setContinuousMode( m_continuous_mode );
    return true;
  } break;

  case 'x': {
    makeRings();
    updateScene();
    sutilCurrentTime( &m_time_of_last_change );
    return true;
  } break;

  case 'c': {
    m_allow_random_changes = !m_allow_random_changes;
    return true;
  } break;

  case 'j': {
    float sun_theta = m_sun_sky.getSunTheta() + 0.05f;
    if( sun_theta > M_PI / 2.0f )
      sun_theta = static_cast<float>( M_PI ) / 2.0f;
    m_sun_sky.setSunTheta( sun_theta );
    updateLights(); 
    return true;
  } break;

  case 'J': {
    float sun_theta = m_sun_sky.getSunTheta() - 0.05f;
    if( sun_theta < 0.0f ) 
      sun_theta = 0.0f;
    m_sun_sky.setSunTheta( sun_theta );
    updateLights(); 
    return true;
  } break;

  case 'k': {
    m_sun_sky.setSunPhi( m_sun_sky.getSunPhi() + 0.05f );
    updateLights(); 
    return true;
  } break;
  case 'K': {
    m_sun_sky.setSunPhi( m_sun_sky.getSunPhi() - 0.05f );
    updateLights(); 
    return true;
  } break;

  case 'l': {
    m_sun_sky.setTurbidity( m_sun_sky.getTurbidity() + 0.05f );
    updateLights(); 
    return true;
  } break;
  case 'L': {
    float turbidity = m_sun_sky.getTurbidity() - 0.05f;
    if( turbidity < 2.0f )
      turbidity = 2.0f;
    m_sun_sky.setTurbidity( turbidity );
    updateLights(); 
    return true;
  } break;

  case 'h': {
    m_sun_scale += 0.0000002f;
    updateLights(); 
    return true;
  } break;
  case 'H': {
    m_sun_scale -= 0.0000002f;
    if( m_sun_scale < 0 )
      m_sun_scale = 0;
    updateLights(); 
    return true;
  } break;

  }

  // A little bit of interactivity
  if( key >= '0' && key <= '9' ) {
    size_t r = key - '0';
    if( r < m_all_rings.Rings.size() ) {
      switch(rand()%3) {
      case 0:
        m_all_rings.Rings[r].ringRot.freezeFrames += 10.0f;
        break;
      case 1:
        m_all_rings.Rings[r].ringRot.rotVel = 1.0f;
        break;
      case 2:
        m_all_rings.Rings[r].ringRot.rotVel = -1.0f;
        break;      
        }
      return true;
    }
  }
 
  m_frame_number = tmp_fn;

  return false;
}

void WhirligigScene::trace( const RayGenCameraData& camera_data )
{
  if( m_camera_changed ) m_frame_number = 0;
  m_camera_changed = false;

  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );
  m_context["frame_number"]->setUint( m_frame_number++ );

  m_sun_sky.setVariables( m_context );

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  if( m_continuous_mode == GLUTDisplay::CDAnimated ) {
    updateScene();
  }

  m_context->launch( getEntryPoint(), buffer_width, buffer_height );
}

void WhirligigScene::doResize( unsigned int width, unsigned int height )
{
  // output buffer handled in SampleScene::resize
  m_context["variance_sum_buffer"]->getBuffer()->setSize( width, height );
  m_context["variance_sum2_buffer"]->getBuffer()->setSize( width, height );
  m_context["num_samples_buffer"]->getBuffer()->setSize( width, height );
  m_context["rnd_seeds"]->getBuffer()->setSize( width, height );
  genRndSeeds( width, height );
}

void WhirligigScene::createGeometry()
{
  ///////////////////////////////////////////////////////////////////
  // Materials

  // Glass material
  Program glass_ch = m_context->createProgramFromPTXFile( ptxpath( "whirligig", "glass.cu" ), "closest_hit_radiance" );
  Program glass_ah = m_context->createProgramFromPTXFile( ptxpath( "whirligig", "glass.cu" ), "any_hit_shadow" );

  for( int i=0; i<2; i++ ) {
    Material glass_matl = m_context->createMaterial();
    glass_matl->setClosestHitProgram( 0, glass_ch );
    glass_matl->setAnyHitProgram( 1, glass_ah );

    // Result of fresnel-schlick term ==0->pure refraction; 1->pure reflection
    glass_matl["importance_cutoff"  ]->setFloat( 0.07f );
    glass_matl["cutoff_color"       ]->setFloat( 0.2f, 0.2f, 0.3f );
    glass_matl["fresnel_exponent"   ]->setFloat( 4.0f );
    glass_matl["fresnel_minimum"    ]->setFloat( 0.1f );
    glass_matl["fresnel_maximum"    ]->setFloat( 1.0f );
    glass_matl["refraction_index"   ]->setFloat( 1.4f );
    glass_matl["refraction_color"   ]->setFloat( 0.98f, 0.5f, 0.7f );
    glass_matl["reflection_color"   ]->setFloat( 1.0f, 1.0f, 1.0f );
    glass_matl["refraction_maxdepth"]->setInt( 7 );
    glass_matl["reflection_maxdepth"]->setInt( 3 );
    float3 extinction = make_float3( .80f, .89f, .75f );
    glass_matl["extinction_constant"]->setFloat( log( extinction.x ), log( extinction.y ), log( extinction.z ) );
    glass_matl["shadow_attenuation"]->setFloat( 0.4f, 0.7f, 0.4f );

    m_materials.push_back( glass_matl );
  }

  m_materials[0]["refraction_color"   ]->setFloat( 0.98f, 0.5f, 0.7f );
  m_materials[0]["shadow_attenuation" ]->setFloat( 0.98f, 0.5f, 0.7f );
  m_materials[1]["refraction_color"   ]->setFloat( 0.5f, 0.98f, 0.7f );
  m_materials[1]["shadow_attenuation" ]->setFloat( 0.5f, 0.98f, 0.7f );

  // Metal material
  Program phong_ch = m_context->createProgramFromPTXFile( ptxpath( "whirligig", "phong.cu" ), "closest_hit_radiance" );
  Program phong_ah = m_context->createProgramFromPTXFile( ptxpath( "whirligig", "phong.cu" ), "any_hit_shadow" );

  Material metal_matl = m_context->createMaterial();
  metal_matl->setClosestHitProgram( 0, phong_ch );
  metal_matl->setAnyHitProgram( 1, phong_ah );
  metal_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
  metal_matl["Kd"]->setFloat( 0.2f, 0.7f, 0.8f );
  metal_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
  metal_matl["phong_exp"]->setFloat( 64 );
  //metal_matl["reflectivity"]->setFloat( 0.5f,  0.5f,  0.5f );
  metal_matl["reflectivity"]->setFloat( 0.0f,  0.0f,  0.0f ); // To speed it up and make it look less busy

  m_materials.push_back( metal_matl );

  ///////////////////////////////////////////////////////////////////
  // Geometry

  std::string sphere_ptx( ptxpath( "whirligig", "sphere_list.cu" ) );

  // Create rings of spheres

  // Count all spheres
  unsigned int nSpheres = m_all_rings.TotalSpheres() + 1u;

  // Allocate sphere buffers
  m_sphere_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, maxSpheres );

  m_sphere_mat_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, maxSpheres );

  m_sphereG = m_context->createGeometry();
  m_sphereG->setPrimitiveCount( nSpheres );
  m_sphereG->setBoundingBoxProgram( m_context->createProgramFromPTXFile( sphere_ptx, "bounds" ) );
  m_sphereG->setIntersectionProgram( m_context->createProgramFromPTXFile( sphere_ptx, "intersect" ) );

  fillInSpheres();

  m_sphereG["sphere_buffer"]->setBuffer( m_sphere_buffer );
  m_sphereG["material_buffer"]->setBuffer( m_sphere_mat_buffer );

  std::vector<GeometryInstance> GIs;
  GIs.push_back( m_context->createGeometryInstance( m_sphereG, m_materials.begin(), m_materials.end() ) );

  GeometryGroup GG = m_context->createGeometryGroup( GIs.begin(), GIs.end() );
  GG->setAcceleration( m_sphere_accel = m_context->createAcceleration( "MedianBvh", "Bvh" ) );

  m_context["top_object"]->set( GG );
  m_context["top_shadower"]->set( GG );
}

void WhirligigScene::fillInSpheres()
{
  unsigned int *sphere_mats = reinterpret_cast<unsigned int *>( m_sphere_mat_buffer->map() );
  float4 *spheres = reinterpret_cast<float4 *>( m_sphere_buffer->map() );

  // Background sphere
  sphere_mats[0] = 2u;
  spheres[0] = make_float4( 0.0f, -50.0f, 0.0f, 48.0f );

  unsigned int i = 1u; // 1 is for the background sphere
  for( size_t r = 0; r < m_all_rings.Rings.size(); r++ ) {
    // Put all the spheres in this ring into the array of sphere centers
    for( size_t s = 0; s < m_all_rings.Rings[r].sphCenters.size(); s++, i++ ) {
      sphere_mats[i] = s==0 ? 1u : 0u;

      float ang = m_all_rings.Rings[r].ringRot.rotPos;
      float3 &cen = m_all_rings.Rings[r].sphCenters[s];
      float3 rot = RotXZ( cen, ang );
      spheres[i] = make_float4( rot.x, rot.y, rot.z, m_all_rings.Rings[r].sphRadius );
    }
  }

  m_sphere_buffer->unmap();
  m_sphere_mat_buffer->unmap();

  m_sphereG->setPrimitiveCount( i );
}

// Apply random spin to a random ring
void WhirligigScene::randRingVel()
{
  size_t r = rand() % m_all_rings.Rings.size();
  m_all_rings.Rings[r].ringRot.useWave = true;
  m_all_rings.Rings[r].ringRot.rotAccelWave = Wave_t( ( FRand()-0.5f ) * 0.02f, FRand() * 0.03f );
}

void WhirligigScene::makeRings()
{
  // Try making random ring configurations until we get one that's cool enough
  size_t nSpheres = 0u;
  do {
    m_all_rings = allRings_t( Wave_t( FRand(), FRand(), FRand(), FRand() ),
      Wave_t( FRand() + 1.0f, FRand(), FRand() * 5.0f + 5.0f, 0.0f ), 0.1f, 10, 90, 8.0f );
    nSpheres = m_all_rings.TotalSpheres();
  } while( nSpheres < 15u || nSpheres > maxSpheres-1u || m_all_rings.Rings.back().ringRadius < 5.0f );

  // Set up how the rings spin
  randRingVel();

  if( m_all_rings.Rings.size() > 5 ) {
      randRingVel();
  }

  // Choose new colors
  size_t m = rand() % ( m_materials.size() - 1u );

  // Choose a color where at least one channel isn't dim
  float3 col;
  const float Dim = 0.75f;
  do {
      col = make_float3( FRand(), FRand(), FRand() );
  } while( col.x < Dim && col.y < Dim && col.z < Dim );

  m_materials[m]["refraction_color"]->setFloat( col );
  m_materials[m]["shadow_attenuation"]->setFloat( col );
}

void WhirligigScene::updateScene()
{
  if( m_time_of_last_change == 0 )
    sutilCurrentTime( &m_time_of_last_change );

  double time_now;
  sutilCurrentTime( &time_now );

  if( time_now - m_time_of_last_change > 5.0 && !GLUTDisplay::isBenchmark() && m_allow_random_changes ) {
    m_time_of_last_change = time_now;

    if( ( rand()%3 ) == 0 ) {
      makeRings();
    } else {
      randRingVel();
    }
  }

  static float t = 0.0f;
  m_all_rings.StepTime( t );

  fillInSpheres();

  m_sphere_accel->markDirty();

  t += 1.0f;
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
    << "  -a  | --adaptive                           Turn on adaptive AA\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  a Cycle frozen, progressive AA, and animation\n"
    << "  x Make a new whirligig\n"
    << "  c Toggle random creation of new whirligigs\n"
    << "0-9 Freeze or flick a ring of spheres\n"
    << "  h Decrease sun brightness\n"
    << "  H Increase sun brightness\n"
    << "  j Decrease sun elevation\n"
    << "  J Increase sun elevation\n"
    << "  k Decrease sun direction\n"
    << "  K Increase sun direction\n"
    << "  l Decrease sky turbidity\n"
    << "  L Increase sky turbidity\n"
    << std::endl;

  if ( doExit ) exit( 1 );
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  GLUTDisplay::contDraw_E continuous_mode = GLUTDisplay::CDAnimated;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--adaptive" || arg == "-a" ) {
      continuous_mode = GLUTDisplay::CDProgressive;
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    WhirligigScene scene( continuous_mode );
    GLUTDisplay::run( "Whirligig", &scene, continuous_mode );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit( 1 );
  }
  return 0;
}
