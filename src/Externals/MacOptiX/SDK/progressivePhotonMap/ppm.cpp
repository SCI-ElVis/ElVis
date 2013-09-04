
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
//  ppm.cpp -- Progressive photon mapping scene
//
//-------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <sutil.h>
#include <ImageLoader.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include "ppm.h"
#include "select.h"
#include "PpmObjLoader.h"
#include "random.h"

using namespace optix;

// Finds the smallest power of 2 greater or equal to x.
inline unsigned int pow2roundup(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x+1;
}

inline float max(float a, float b)
{
  return a > b ? a : b;
}

inline RT_HOSTDEVICE int max_component(float3 a)
{
  if(a.x > a.y) {
    if(a.x > a.z) {
      return 0;
    } else {
      return 2;
    }
  } else {
    if(a.y > a.z) {
      return 1;
    } else {
      return 2;
    }
  }
}

float3 sphericalToCartesian( float theta, float phi )
{
  float cos_theta = cosf( theta );
  float sin_theta = sinf( theta );
  float cos_phi = cosf( phi );
  float sin_phi = sinf( phi );
  float3 v;
  v.x = cos_phi * sin_theta;
  v.z = sin_phi * sin_theta;
  v.y = cos_theta;
  return v;
}

enum SplitChoice {
  RoundRobin,
  HighestVariance,
  LongestDim
};

//-----------------------------------------------------------------------------
//
// Whitted Scene
//
//-----------------------------------------------------------------------------

class ProgressivePhotonScene : public SampleScene
{
public:
  ProgressivePhotonScene() : SampleScene()
    , m_frame_number( 0 )
    , m_display_debug_buffer( false )
    , m_print_timings ( false )
    , m_cornell_box( false )
    , m_light_phi( 2.19f )
    , m_light_theta( 1.15f )
    , m_split_choice(LongestDim)
  {}

  // From SampleScene
  void   initScene( InitialCameraData& camera_data );
  bool   keyPressed(unsigned char key, int x, int y);
  void   trace( const RayGenCameraData& camera_data );
  void   doResize( unsigned int width, unsigned int height );
  Buffer getOutputBuffer();

  void setSceneCornellBox() { m_cornell_box = true; }
  void setSceneOBJ()        { m_cornell_box = false; }
  void printTimings()       { m_print_timings = true; }
  void displayDebugBuffer() { m_display_debug_buffer = true; }
private:
  void createPhotonMap();
  void loadObjGeometry( const std::string& filename, optix::Aabb& bbox );
  void createCornellBoxGeometry();
  GeometryInstance createParallelogram( const float3& anchor,
                                        const float3& offset1,
                                        const float3& offset2,
                                        const float3& color );

  enum ProgramEnum {
    rtpass,
    ppass,
    gather,
    numPrograms
  };

  unsigned int  m_frame_number;
  bool          m_display_debug_buffer;
  bool          m_print_timings;
  bool          m_cornell_box;
  Program       m_pgram_bounding_box;
  Program       m_pgram_intersection;
  Material      m_material;
  Buffer        m_display_buffer;
  Buffer        m_photons;
  Buffer        m_photon_map;
  Buffer        m_debug_buffer;
  float         m_light_phi;
  float         m_light_theta;
  unsigned int  m_iteration_count;
  unsigned int  m_photon_map_size;
  SplitChoice   m_split_choice;
  PPMLight      m_light;

  const static unsigned int WIDTH;
  const static unsigned int HEIGHT;
  const static unsigned int MAX_PHOTON_COUNT;
  const static unsigned int PHOTON_LAUNCH_WIDTH;
  const static unsigned int PHOTON_LAUNCH_HEIGHT;
  const static unsigned int NUM_PHOTONS;

};

const unsigned int ProgressivePhotonScene::WIDTH  = 768u;
const unsigned int ProgressivePhotonScene::HEIGHT = 768u;
const unsigned int ProgressivePhotonScene::MAX_PHOTON_COUNT = 2u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH = 256u;
const unsigned int ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT = 256u;
const unsigned int ProgressivePhotonScene::NUM_PHOTONS = (ProgressivePhotonScene::PHOTON_LAUNCH_WIDTH *
                                                          ProgressivePhotonScene::PHOTON_LAUNCH_HEIGHT *
                                                          ProgressivePhotonScene::MAX_PHOTON_COUNT);


bool ProgressivePhotonScene::keyPressed(unsigned char key, int x, int y)
{
  float step_size = 0.01f;
  bool light_changed = false;;
  switch (key)
  {
    case 'd':
      m_light_phi += step_size;
      if( m_light_phi >  M_PIf * 2.0f ) m_light_phi -= M_PIf * 2.0f;
      light_changed = true;
      break;
    case 'a':
      m_light_phi -= step_size;
      if( m_light_phi <  0.0f ) m_light_phi += M_PIf * 2.0f;
      light_changed = true;
      break;
    case 's':
      std::cerr << "new theta: " << m_light_theta + step_size << " max: " << M_PIf / 2.0f  << std::endl;
      m_light_theta = fminf( m_light_theta + step_size, M_PIf / 2.0f );
      light_changed = true;
      break;
    case 'w':
      std::cerr << "new theta: " << m_light_theta - step_size << " min: 0.0f " << std::endl;
      m_light_theta = fmaxf( m_light_theta - step_size, 0.0f );
      light_changed = true;
      break;
  }

  if( light_changed && !m_cornell_box ) {
    std::cerr << " theta: " << m_light_theta << "  phi: " << m_light_phi << std::endl;
    m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
    m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
    m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
    signalCameraChanged(); 
    return true;
  }

  return false;
}

void ProgressivePhotonScene::initScene( InitialCameraData& camera_data )
{
  // There's a performance advantage to using a device that isn't being used as a display.
  // We'll take a guess and pick the second GPU if the second one has the same compute
  // capability as the first.
  int deviceId = 0;
  int computeCaps[2];
  if (RTresult code = rtDeviceGetAttribute(0, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCaps))
    throw Exception::makeException(code, 0);
  for(unsigned int index = 1; index < Context::getDeviceCount(); ++index) {
    int computeCapsB[2];
    if (RTresult code = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY, sizeof(computeCaps), &computeCapsB))
      throw Exception::makeException(code, 0);
    if (computeCaps[0] == computeCapsB[0] && computeCaps[1] == computeCapsB[1]) {
      deviceId = index;
      break;
    }
  }
  m_context->setDevices(&deviceId, &deviceId+1);

  m_context->setRayTypeCount( 3 );
  m_context->setEntryPointCount( numPrograms );
  m_context->setStackSize( 960 );

  m_context["max_depth"]->setUint(3);
  m_context["max_photon_count"]->setUint(MAX_PHOTON_COUNT);
  m_context["scene_epsilon"]->setFloat( 1.e-1f );
  m_context["alpha"]->setFloat( 0.7f );
  m_context["total_emitted"]->setFloat( 0.0f );
  m_context["frame_number"]->setFloat( 0.0f );
  m_context["use_debug_buffer"]->setUint( m_display_debug_buffer ? 1 : 0 );

  // Display buffer
  m_display_buffer = createOutputBuffer(RT_FORMAT_FLOAT4, WIDTH, HEIGHT);
  m_context["output_buffer"]->set( m_display_buffer );

  // Debug output buffer
  m_debug_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
  m_debug_buffer->setFormat( RT_FORMAT_FLOAT4 );
  m_debug_buffer->setSize( WIDTH, HEIGHT );
  m_context["debug_buffer"]->set( m_debug_buffer );

  // RTPass output buffer
  Buffer output_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT );
  output_buffer->setFormat( RT_FORMAT_USER );
  output_buffer->setElementSize( sizeof( HitRecord ) );
  output_buffer->setSize( WIDTH, HEIGHT );
  m_context["rtpass_output_buffer"]->set( output_buffer );

  // RTPass pixel sample buffers
  Buffer image_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT2, WIDTH, HEIGHT );
  m_context["image_rnd_seeds"]->set( image_rnd_seeds );
  uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
  for ( unsigned int i = 0; i < WIDTH*HEIGHT; ++i )  
    seeds[i] = random2u();
  image_rnd_seeds->unmap();

  // RTPass ray gen program
  {
    std::string ptx_path = ptxpath( "progressivePhotonMap", "ppm_rtpass.cu" );
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_camera" );
    m_context->setRayGenerationProgram( rtpass, ray_gen_program );

    // RTPass exception/miss programs
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "rtpass_exception" );
    m_context->setExceptionProgram( rtpass, exception_program );
    m_context["rtpass_bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
    m_context->setMissProgram( rtpass, m_context->createProgramFromPTXFile( ptx_path, "rtpass_miss" ) );
    m_context["rtpass_bg_color"]->setFloat( make_float3( 0.34f, 0.55f, 0.85f ) );
  }

  // Set up camera
  camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -800.0f ), // eye
                                   make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
                                   make_float3( 0.0f, 1.0f,  0.0f ),       // up
                                   35.0f );                                // vfov

  // Declare these so validation will pass
  m_context["rtpass_eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["rtpass_U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["rtpass_V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["rtpass_W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  // Photon pass
  m_photons = m_context->createBuffer( RT_BUFFER_OUTPUT );
  m_photons->setFormat( RT_FORMAT_USER );
  m_photons->setElementSize( sizeof( PhotonRecord ) );
  m_photons->setSize( NUM_PHOTONS );
  m_context["ppass_output_buffer"]->set( m_photons );


  {
    std::string ptx_path = ptxpath( "progressivePhotonMap", "ppm_ppass.cu");
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "ppass_camera" );
    m_context->setRayGenerationProgram( ppass, ray_gen_program );

    Buffer photon_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT,
                                                      RT_FORMAT_UNSIGNED_INT2,
                                                      PHOTON_LAUNCH_WIDTH,
                                                      PHOTON_LAUNCH_HEIGHT );
    uint2* seeds = reinterpret_cast<uint2*>( photon_rnd_seeds->map() );
    for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i )
      seeds[i] = random2u();
    photon_rnd_seeds->unmap();
    m_context["photon_rnd_seeds"]->set( photon_rnd_seeds );

  }

  // Gather phase
  {
    std::string ptx_path = ptxpath( "progressivePhotonMap", "ppm_gather.cu" );
    Program gather_program = m_context->createProgramFromPTXFile( ptx_path, "gather" );
    m_context->setRayGenerationProgram( gather, gather_program );
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "gather_exception" );
    m_context->setExceptionProgram( gather, exception_program );
  
    m_photon_map_size = pow2roundup( NUM_PHOTONS ) - 1;
    m_photon_map = m_context->createBuffer( RT_BUFFER_INPUT );
    m_photon_map->setFormat( RT_FORMAT_USER );
    m_photon_map->setElementSize( sizeof( PhotonRecord ) );
    m_photon_map->setSize( m_photon_map_size );
    m_context["photon_map"]->set( m_photon_map );
  }

  // Populate scene hierarchy
  if( !m_cornell_box ) {
    optix::Aabb aabb;
    loadObjGeometry( "wedding-band.obj", aabb);
    camera_data = InitialCameraData( make_float3( -235.0f, 220.0f, 0.0f ), // eye
                                     make_float3( 0.0f, 0.0f, 0.0f ),      // lookat
                                     make_float3( 0.0f, 1.0f,  0.0f ),     // up
                                     35.0f );                              // vfov
    m_light.is_area_light = 0; 
    m_light.position  = 1000.0f * sphericalToCartesian( m_light_theta, m_light_phi );
    //light.position = make_float3( 600.0f, 500.0f, 700.0f );
    m_light.direction = normalize( make_float3( 0.0f, 0.0f, 0.0f )  - m_light.position );
    m_light.radius    = 5.0f *0.01745329252f;
    m_light.power     = make_float3( 0.5e4f, 0.5e4f, 0.5e4f );
    m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
    m_context["rtpass_default_radius2"]->setFloat( 0.25f);
    m_context["ambient_light"]->setFloat( 0.1f, 0.1f, 0.1f);
    std::string full_path = std::string( sutilSamplesDir() ) + "/tutorial/data/CedarCity.hdr";
    const float3 default_color = make_float3( 0.8f, 0.88f, 0.97f );
    m_context["envmap"]->setTextureSampler( loadTexture( m_context, full_path, default_color) );
  } else {

    createCornellBoxGeometry();
    // Set up camera
    camera_data = InitialCameraData( make_float3( 278.0f, 273.0f, -850.0f ), // eye
                                     make_float3( 278.0f, 273.0f, 0.0f ),    // lookat
                                     make_float3( 0.0f, 1.0f,  0.0f ),       // up
                                     35.0f );                                // vfov

    m_light.is_area_light = 1; 
    m_light.anchor = make_float3( 343.0f, 548.6f, 227.0f);
    m_light.v1     = make_float3( 0.0f, 0.0f, 105.0f);
    m_light.v2     = make_float3( -130.0f, 0.0f, 0.0f);
    m_light.direction = normalize(cross( m_light.v1, m_light.v2 ) ); 
    m_light.power  = make_float3( 0.5e6f, 0.4e6f, 0.2e6f );
    m_context["light"]->setUserData( sizeof(PPMLight), &m_light );
    m_context["rtpass_default_radius2"]->setFloat( 400.0f);
    m_context["ambient_light"]->setFloat( 0.0f, 0.0f, 0.0f);
    const float3 default_color = make_float3(0.0f, 0.0f, 0.0f);
    m_context["envmap"]->setTextureSampler( loadTexture( m_context, "", default_color) );
  }
    

  // Prepare to run
  m_context->validate();
  m_context->compile();
}

Buffer ProgressivePhotonScene::getOutputBuffer()
{
  return m_display_buffer;
}

inline uchar4 makeColor( const float3& c )
{
  uchar4 pixel;
  pixel.x = static_cast<unsigned char>( fmaxf( fminf( c.z, 1.0f ), 0.0f ) * 255.99f );
  pixel.y = static_cast<unsigned char>( fmaxf( fminf( c.y, 1.0f ), 0.0f ) * 255.99f );
  pixel.z = static_cast<unsigned char>( fmaxf( fminf( c.x, 1.0f ), 0.0f ) * 255.99f );
  pixel.w = 0; 
  return pixel;
}


bool photonCmpX( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.x < r2->position.x; }
bool photonCmpY( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.y < r2->position.y; }
bool photonCmpZ( PhotonRecord* r1, PhotonRecord* r2 ) { return r1->position.z < r2->position.z; }


void buildKDTree( PhotonRecord** photons, int start, int end, int depth, PhotonRecord* kd_tree, int current_root,
                  SplitChoice split_choice, float3 bbmin, float3 bbmax)
{
  // If we have zero photons, this is a NULL node
  if( end - start == 0 ) {
    kd_tree[current_root].axis = PPM_NULL;
    kd_tree[current_root].energy = make_float3( 0.0f );
    return;
  }

  // If we have a single photon
  if( end - start == 1 ) {
    photons[start]->axis = PPM_LEAF;
    kd_tree[current_root] = *(photons[start]);
    return;
  }

  // Choose axis to split on
  int axis;
  switch(split_choice) {
  case RoundRobin:
    {
      axis = depth%3;
    }
    break;
  case HighestVariance:
    {
      float3 mean  = make_float3( 0.0f ); 
      float3 diff2 = make_float3( 0.0f );
      for(int i = start; i < end; ++i) {
        float3 x     = photons[i]->position;
        float3 delta = x - mean;
        float3 n_inv = make_float3( 1.0f / ( static_cast<float>( i - start ) + 1.0f ) );
        mean = mean + delta * n_inv;
        diff2 += delta*( x - mean );
      }
      float3 n_inv = make_float3( 1.0f / ( static_cast<float>(end-start) - 1.0f ) );
      float3 variance = diff2 * n_inv;
      axis = max_component(variance);
    }
    break;
  case LongestDim:
    {
      float3 diag = bbmax-bbmin;
      axis = max_component(diag);
    }
    break;
  default:
    axis = -1;
    std::cerr << "Unknown SplitChoice " << split_choice << " at "<<__FILE__<<":"<<__LINE__<<"\n";
    exit(2);
    break;
  }

  int median = (start+end) / 2;
  PhotonRecord** start_addr = &(photons[start]);
#if 0
  switch( axis ) {
  case 0:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpX );
    photons[median]->axis = PPM_X;
    break;
  case 1:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpY );
    photons[median]->axis = PPM_Y;
    break;
  case 2:
    std::nth_element( start_addr, start_addr + median-start, start_addr + end-start, photonCmpZ );
    photons[median]->axis = PPM_Z;
    break;
  }
#else
  switch( axis ) {
  case 0:
    select<PhotonRecord*, 0>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_X;
    break;
  case 1:
    select<PhotonRecord*, 1>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_Y;
    break;
  case 2:
    select<PhotonRecord*, 2>( start_addr, 0, end-start-1, median-start );
    photons[median]->axis = PPM_Z;
    break;
  }
#endif
  float3 rightMin = bbmin;
  float3 leftMax  = bbmax;
  if(split_choice == LongestDim) {
    float3 midPoint = (*photons[median]).position;
    switch( axis ) {
      case 0:
        rightMin.x = midPoint.x;
        leftMax.x  = midPoint.x;
        break;
      case 1:
        rightMin.y = midPoint.y;
        leftMax.y  = midPoint.y;
        break;
      case 2:
        rightMin.z = midPoint.z;
        leftMax.z  = midPoint.z;
        break;
    }
  }

  kd_tree[current_root] = *(photons[median]);
  buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, split_choice, bbmin,  leftMax );
  buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, split_choice, rightMin, bbmax );
}


void ProgressivePhotonScene::createPhotonMap()
{
  PhotonRecord* photons_data    = reinterpret_cast<PhotonRecord*>( m_photons->map() );
  PhotonRecord* photon_map_data = reinterpret_cast<PhotonRecord*>( m_photon_map->map() );

  for( unsigned int i = 0; i < m_photon_map_size; ++i ) {
    photon_map_data[i].energy = make_float3( 0.0f );
  }

  // Push all valid photons to front of list
  unsigned int valid_photons = 0;
  PhotonRecord** temp_photons = new PhotonRecord*[NUM_PHOTONS];
  for( unsigned int i = 0; i < NUM_PHOTONS; ++i ) {
    if( fmaxf( photons_data[i].energy ) > 0.0f ) {
      temp_photons[valid_photons++] = &photons_data[i];
    }
  }
  if ( m_display_debug_buffer ) {
    std::cerr << " ** valid_photon/NUM_PHOTONS =  " 
              << valid_photons<<"/"<<NUM_PHOTONS
              <<" ("<<valid_photons/static_cast<float>(NUM_PHOTONS)<<")\n";
  }

  // Make sure we arent at most 1 less than power of 2
  valid_photons = valid_photons >= m_photon_map_size ? m_photon_map_size : valid_photons;

  float3 bbmin = make_float3(0.0f);
  float3 bbmax = make_float3(0.0f);
  if( m_split_choice == LongestDim ) {
    bbmin = make_float3(  std::numeric_limits<float>::max() );
    bbmax = make_float3( -std::numeric_limits<float>::max() );
    // Compute the bounds of the photons
    for(unsigned int i = 0; i < valid_photons; ++i) {
      float3 position = (*temp_photons[i]).position;
      bbmin = fminf(bbmin, position);
      bbmax = fmaxf(bbmax, position);
    }
  }

  // Now build KD tree
  buildKDTree( temp_photons, 0, valid_photons, 0, photon_map_data, 0, m_split_choice, bbmin, bbmax );

  delete[] temp_photons;
  m_photon_map->unmap();
  m_photons->unmap();
}

void ProgressivePhotonScene::trace( const RayGenCameraData& camera_data )
{
  Buffer output_buffer = m_context["rtpass_output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  output_buffer->getSize( buffer_width, buffer_height );

  m_frame_number = m_camera_changed ? 0u : m_frame_number+1;
  m_context["frame_number"]->setFloat( static_cast<float>(m_frame_number) );
  if ( m_camera_changed ) {
    m_camera_changed = false;
    m_context["rtpass_eye"]->setFloat( camera_data.eye );
    m_context["rtpass_U"]->setFloat( camera_data.U );
    m_context["rtpass_V"]->setFloat( camera_data.V );
    m_context["rtpass_W"]->setFloat( camera_data.W );
  
    // Trace viewing rays
    if (m_print_timings) std::cerr << "Starting RT pass ... ";
    std::cerr.flush();
    double t0, t1;
    sutilCurrentTime( &t0 );
    m_context->launch( rtpass,
                      static_cast<unsigned int>(buffer_width),
                      static_cast<unsigned int>(buffer_height) );
    sutilCurrentTime( &t1 );
    if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;
    m_context["total_emitted"]->setFloat(  0.0f );
    m_iteration_count=1;
  }

  // Trace photons
  if (m_print_timings) std::cerr << "Starting photon pass   ... ";
  Buffer photon_rnd_seeds = m_context["photon_rnd_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( photon_rnd_seeds->map() );
  for ( unsigned int i = 0; i < PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT; ++i )
    seeds[i] = random2u();
  photon_rnd_seeds->unmap();
  double t0, t1;
  sutilCurrentTime( &t0 );
  m_context->launch( ppass,
                    static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
                    static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT) );
  // By computing the total number of photons as an unsigned long long we avoid 32 bit
  // floating point addition errors when the number of photons gets sufficiently large
  // (the error of adding two floating point numbers when the mantissa bits no longer
  // overlap).
  m_context["total_emitted"]->setFloat( static_cast<float>((unsigned long long)m_iteration_count*PHOTON_LAUNCH_WIDTH*PHOTON_LAUNCH_HEIGHT) );
  sutilCurrentTime( &t1 );
  if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

  // Build KD tree 
  if (m_print_timings) std::cerr << "Starting kd_tree build ... ";
  sutilCurrentTime( &t0 );
  createPhotonMap();
  sutilCurrentTime( &t1 );
  if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

  // Shade view rays by gathering photons
  if (m_print_timings) std::cerr << "Starting gather pass   ... ";
  sutilCurrentTime( &t0 );
  m_context->launch( gather,
                    static_cast<unsigned int>(buffer_width),
                    static_cast<unsigned int>(buffer_height) );
  sutilCurrentTime( &t1 );
  if (m_print_timings) std::cerr << "finished. " << t1 - t0 << std::endl;

  // Debug output
  if( m_display_debug_buffer ) {
    sutilCurrentTime( &t0 );
    float4* debug_data = reinterpret_cast<float4*>( m_debug_buffer->map() );
    Buffer hit_records = m_context["rtpass_output_buffer"]->getBuffer();
    HitRecord* hit_record_data = reinterpret_cast<HitRecord*>( hit_records->map() );
    float4 avg  = make_float4( 0.0f );
    float4 minv = make_float4( std::numeric_limits<float>::max() );
    float4 maxv = make_float4( 0.0f );
    float counter = 0.0f;
    for( unsigned int j = 0; j < buffer_height; ++j ) {
      for( unsigned int i = 0; i < buffer_width; ++i ) {
        /*
        if( i < 10 && j < 10 && 0) {
          fprintf( stderr, " %08.4f %08.4f %08.4f %08.4f\n", debug_data[j*buffer_width+i].x,
                                                             debug_data[j*buffer_width+i].y,
                                                             debug_data[j*buffer_width+i].z,
                                                             debug_data[j*buffer_width+i].w );
        }
        */

        
        if( hit_record_data[j*buffer_width+i].flags & PPM_HIT ) {
          float4 val = debug_data[j*buffer_width+i];
          avg += val;
          minv = fminf(minv, val);
          maxv = fmaxf(maxv, val);
          counter += 1.0f;
        }
      }
    }
    m_debug_buffer->unmap();
    hit_records->unmap();

    avg = avg / counter; 
    sutilCurrentTime( &t1 );
    if (m_print_timings) std::cerr << "Stat collection time ...           " << t1 - t0 << std::endl;
    std::cerr << "(min, max, average):"
      << " loop iterations: ( "
      << minv.x << ", "
      << maxv.x << ", "
      << avg.x << " )"
      << " radius: ( "
      << minv.y << ", "
      << maxv.y << ", "
      << avg.y << " )"
      << " N: ( "
      << minv.z << ", "
      << maxv.z << ", "
      << avg.z << " )"
      << " M: ( "
      << minv.w << ", "
      << maxv.w << ", "
      << avg.w << " )";
    std::cerr << ", total_iterations = "<<m_iteration_count;
    std::cerr << std::endl;
  }
  m_iteration_count++;
}


void ProgressivePhotonScene::doResize( unsigned int width, unsigned int height )
{
  // display buffer resizing handled in base class
  m_context["rtpass_output_buffer"]->getBuffer()->setSize( width, height );
  m_context["output_buffer"       ]->getBuffer()->setSize( width, height );
  m_context["image_rnd_seeds"     ]->getBuffer()->setSize( width, height );
  m_context["debug_buffer"        ]->getBuffer()->setSize( width, height );
  
  Buffer image_rnd_seeds = m_context["image_rnd_seeds"]->getBuffer();
  uint2* seeds = reinterpret_cast<uint2*>( image_rnd_seeds->map() );
  for ( unsigned int i = 0; i < width*height; ++i )  
    seeds[i] = random2u();
  image_rnd_seeds->unmap();
}

GeometryInstance ProgressivePhotonScene::createParallelogram( const float3& anchor,
                                                              const float3& offset1,
                                                              const float3& offset2,
                                                              const float3& color )
{
  Geometry parallelogram = m_context->createGeometry();
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setIntersectionProgram( m_pgram_intersection );
  parallelogram->setBoundingBoxProgram( m_pgram_bounding_box );

  float3 normal = normalize( cross( offset1, offset2 ) );
  float d       = dot( normal, anchor );
  float4 plane  = make_float4( normal, d );

  float3 v1 = offset1 / dot( offset1, offset1 );
  float3 v2 = offset2 / dot( offset2, offset2 );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["anchor"]->setFloat( anchor );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );

  GeometryInstance gi = m_context->createGeometryInstance( parallelogram,
      &m_material,
      &m_material+1 );
  gi["Kd"]->setFloat( color );
  gi["Ks"]->setFloat( 0.0f, 0.0f, 0.0f );
  gi["use_grid"]->setUint( 0u );
  gi["grid_color"]->setFloat( make_float3( 0.0f ) );
  gi["emitted"]->setFloat( 0.0f, 0.0f, 0.0f );
  return gi;
}


void ProgressivePhotonScene::loadObjGeometry( const std::string& filename, optix::Aabb& bbox )
{
  GeometryGroup geometry_group = m_context->createGeometryGroup();
  std::string full_path = std::string( sutilSamplesDir() ) + "/progressivePhotonMap/wedding-band.obj";
  PpmObjLoader loader( full_path, m_context, geometry_group );
  loader.load();
  bbox = loader.getSceneBBox();

  m_context["top_object"]->set( geometry_group );
  m_context["top_shadower"]->set( geometry_group );
}


void ProgressivePhotonScene::createCornellBoxGeometry()
{
  // Set up material
  m_material = m_context->createMaterial();
  m_material->setClosestHitProgram( rtpass, m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_rtpass.cu"),
                                   "rtpass_closest_hit") );
  m_material->setClosestHitProgram( ppass,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_ppass.cu"),
                                   "ppass_closest_hit") );
  m_material->setAnyHitProgram(     gather,  m_context->createProgramFromPTXFile( ptxpath( "progressivePhotonMap", "ppm_gather.cu"),
                                   "gather_any_hit") );


  std::string ptx_path = ptxpath( "progressivePhotonMap", "parallelogram.cu" );
  m_pgram_bounding_box = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
  m_pgram_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersect" );


  // create geometry instances
  std::vector<GeometryInstance> gis;

  const float3 white = make_float3( 0.8f, 0.8f, 0.8f );
  const float3 green = make_float3( 0.05f, 0.8f, 0.05f );
  const float3 red   = make_float3( 0.8f, 0.05f, 0.05f );
  const float3 black = make_float3( 0.0f, 0.0f, 0.0f );
  const float3 light = make_float3( 15.0f, 15.0f, 5.0f );

  // Floor
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ),
                                      white ) );

  // Ceiling
  gis.push_back( createParallelogram( make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      white ) );

  // Back wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 559.2f),
                                      make_float3( 0.0f, 548.8f, 0.0f),
                                      make_float3( 556.0f, 0.0f, 0.0f),
                                      white ) );

  // Right wall
  gis.push_back( createParallelogram( make_float3( 0.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      green ) );

  // Left wall
  gis.push_back( createParallelogram( make_float3( 556.0f, 0.0f, 0.0f ),
                                      make_float3( 0.0f, 0.0f, 559.2f ),
                                      make_float3( 0.0f, 548.8f, 0.0f ),
                                      red ) );

  // Short block
  gis.push_back( createParallelogram( make_float3( 130.0f, 165.0f, 65.0f),
                                      make_float3( -48.0f, 0.0f, 160.0f),
                                      make_float3( 160.0f, 0.0f, 49.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 290.0f, 0.0f, 114.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( -50.0f, 0.0f, 158.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 130.0f, 0.0f, 65.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( 160.0f, 0.0f, 49.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 82.0f, 0.0f, 225.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( 48.0f, 0.0f, -160.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 240.0f, 0.0f, 272.0f),
                                      make_float3( 0.0f, 165.0f, 0.0f),
                                      make_float3( -158.0f, 0.0f, -47.0f),
                                      white ) );

  // Tall block
  gis.push_back( createParallelogram( make_float3( 423.0f, 330.0f, 247.0f),
                                      make_float3( -158.0f, 0.0f, 49.0f),
                                      make_float3( 49.0f, 0.0f, 159.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 423.0f, 0.0f, 247.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( 49.0f, 0.0f, 159.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 472.0f, 0.0f, 406.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( -158.0f, 0.0f, 50.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 314.0f, 0.0f, 456.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( -49.0f, 0.0f, -160.0f),
                                      white ) );
  gis.push_back( createParallelogram( make_float3( 265.0f, 0.0f, 296.0f),
                                      make_float3( 0.0f, 330.0f, 0.0f),
                                      make_float3( 158.0f, 0.0f, -49.0f),
                                      white ) );

  // Light
  gis.push_back( createParallelogram( make_float3( 343.0f, 548.7f, 227.0f),
                                      make_float3( 0.0f, 0.0f, 105.0f),
                                      make_float3( -130.0f, 0.0f, 0.0f),
                                      black) );
  gis.back()["emitted"]->setFloat( light );


  // Create geometry group
  GeometryGroup geometry_group = m_context->createGeometryGroup();
  geometry_group->setChildCount( static_cast<unsigned int>( gis.size() ) );
  for ( unsigned int i = 0; i < gis.size(); ++i )
    geometry_group->setChild( i, gis[i] );
  geometry_group->setAcceleration( m_context->createAcceleration("Bvh","Bvh") );

  m_context["top_object"]->set( geometry_group );
  m_context["top_shadower"]->set( geometry_group );
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
    << "  -c  | --cornell-box                        Display Cornell Box scene\n"
    << "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
#ifndef RELEASE_PUBLIC
    << "  -pt | --print-timings                      Print timing information\n"
    << " -ddb | --display-debug-buffer               Display the debug buffer information\n"
#endif
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  w Move light up\n"
    << "  a Move light left\n"
    << "  s Move light down\n"
    << "  d Move light right\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  bool print_timings = false;
  bool display_debug_buffer = false;
  bool cornell_box = false;
  float timeout = -1.0f;

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "--print-timings" || arg == "-pt" ) {
      print_timings = true;
    } else if ( arg == "--display-debug-buffer" || arg == "-ddb" ) {
      display_debug_buffer = true;
    } else if ( arg == "--cornell-box" || arg == "-c" ) {
      cornell_box = true;
    } else if ( arg == "--timeout" || arg == "-t" ) {
      if(++i < argc) {
        timeout = static_cast<float>(atof(argv[i]));
      } else {
        std::cerr << "Missing argument to "<<arg<<"\n";
        printUsageAndExit(argv[0]);
      }
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    ProgressivePhotonScene scene;
    if (print_timings) scene.printTimings();
    if (display_debug_buffer) scene.displayDebugBuffer();
    if (cornell_box ) scene.setSceneCornellBox();
    GLUTDisplay::setProgressiveDrawingTimeout(timeout);
    GLUTDisplay::setUseSRGB(true);
    GLUTDisplay::run( "ProgressivePhotonScene", &scene, GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
