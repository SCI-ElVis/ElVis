
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

//-----------------------------------------------------------------------------
//
//  ocean.cpp: Demonstrate cuda-optix interop via ocean demo.  Based on CUDA
//             SDK sample oceanFFT.
//
//-----------------------------------------------------------------------------

#include <fstream>
#include <iostream>
#include <cfloat>
#include <cstdlib>
#include <cstring>

#include <optixu/optixpp.h>

#include <GLUTDisplay.h>
#include <sutil.h>
#include <SunSky.h>

#include "commonStructs.h"
#include "random.h"

#include <cufft.h>
#include <cuda_runtime.h>


using namespace optixu;


//------------------------------------------------------------------------------
//
// Helpers for checking cuda and cufft return codes
//
//------------------------------------------------------------------------------

#define cutilSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() Runtime error: file <%s>, line %i : %s.\n",
            file, line, cudaGetErrorString( err) );
    exit(-1);
  }
}

#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)

inline void __cufftSafeCall( cufftResult err, const char *file, const int line )
{
  if( CUFFT_SUCCESS != err) {
    std::string mssg = err == CUFFT_INVALID_PLAN     ? "invalid plan"    :
                       err == CUFFT_ALLOC_FAILED     ? "alloc failed"    :
                       err == CUFFT_INVALID_TYPE     ? "invalid type"    :
                       err == CUFFT_INVALID_VALUE    ? "invalid value"   :
                       err == CUFFT_INTERNAL_ERROR   ? "internal error"  :
                       err == CUFFT_EXEC_FAILED      ? "exec failed"     :
                       err == CUFFT_SETUP_FAILED     ? "setup failed"    :
                       err == CUFFT_INVALID_SIZE     ? "invalid size"    :
                       "bad error code!!!!";

    fprintf(stderr, "cufftSafeCall() CUFFT error '%s' in file <%s>, line %i.\n",
            mssg.c_str(), file, line);
    exit(-1);
  }
}


//-----------------------------------------------------------------------------
// 
// Ocean Scene
//
//-----------------------------------------------------------------------------

class OceanScene : public SampleScene
{
public:
  OceanScene()
  : SampleScene(),
    m_output_buffer(0),
    m_data_buffer(0),
    m_normal_buffer(0),
    m_h0_buffer(0),
    m_ht_buffer(0),
    m_frame(0),
    m_animate(true),
    m_tonemap(true),

    m_anim_time_scale( 0.25f ),
    m_anim_time( 0.0 ),
    m_last_time( 0.0 ),
    m_patch_size( 100.0f ),
    m_h_h0( NULL ),
    m_d_h0( NULL ),
    m_d_ht( NULL ),
    m_d_ht_real( NULL )
  { }
  
  // From SampleScene
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   doResize( unsigned int width, unsigned int height );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();
  bool           keyPressed(unsigned char key, int x, int y);
  
  void createGeometry();
  void updateHeightfield();

private:
  void genRndSeeds( unsigned int width, unsigned int height );

  Buffer       m_accum_buffer;
  Buffer       m_seed_buffer;
  Buffer       m_output_buffer;
  Buffer       m_data_buffer;
  Buffer       m_normal_buffer;
  Buffer       m_h0_buffer;
  Buffer       m_ht_buffer;
  Buffer       m_ik_ht_buffer;
  int          m_frame;
  bool         m_animate;
  bool         m_tonemap;


  cufftHandle  m_fft_plan;
  float        m_anim_time_scale;
  double       m_anim_time;
  double       m_last_time;
  float        m_patch_size;
  float2*      m_h_h0;
  float2*      m_d_h0;
  float2*      m_d_ht;
  float2*      m_h_ht;
  float*       m_d_ht_real;
  float*       m_h_ht_real;

  PreethamSunSky  m_sun_sky;


  static float phillips(float Kx, float Ky, float Vdir, float V, float A);
  void  generateH0( float2* h_h0 );

  static unsigned int WIDTH;
  static unsigned int HEIGHT;
  
  static unsigned int HEIGHTFIELD_WIDTH;
  static unsigned int HEIGHTFIELD_HEIGHT;
  
  static unsigned int FFT_WIDTH;
  static unsigned int FFT_HEIGHT;
};

unsigned int OceanScene::WIDTH  = 1024u;
unsigned int OceanScene::HEIGHT = 768u;

unsigned int OceanScene::HEIGHTFIELD_WIDTH  = 1024;
unsigned int OceanScene::HEIGHTFIELD_HEIGHT = 1024;
unsigned int OceanScene::FFT_WIDTH          = HEIGHTFIELD_WIDTH/2 + 1;
unsigned int OceanScene::FFT_HEIGHT         = HEIGHTFIELD_HEIGHT;

bool OceanScene::keyPressed(unsigned char key, int x, int y)
{
  switch(key) {
    case ' ':{
      m_animate = !m_animate;
      if( !m_animate )
        m_context[ "jitter_factor" ]->setFloat( 1.0f );
      else
        m_context[ "jitter_factor" ]->setFloat( 0.0f );
        
      return true;
    } break;
    case 't':{
      m_tonemap= !m_tonemap;
      return true;
    } break;
  }

  return false;
}


// helper function to extract device pointer
template <class T>
T rtGetBDP( Context& context, RTbuffer buf, int optixDeviceIndex )
{
  void* bdp;
  RTresult res = rtBufferGetDevicePointer(buf, optixDeviceIndex, &bdp);
  if ( RT_SUCCESS != res )
  {
    sutilHandleErrorNoExit( context->get(), res, __FILE__, __LINE__ );
  }
  return (T) bdp;
}


// helper function to obtain the CUDA device ordinal of a given OptiX device
int GetOptixDeviceOrdinal( const Context& context, unsigned int optixDeviceIndex )
{
  unsigned int numOptixDevices = context->getEnabledDeviceCount();
  std::vector<int> devices = context->getEnabledDevices();
  int ordinal;
  if ( optixDeviceIndex < numOptixDevices )
  {
    context->getDeviceAttribute( devices[optixDeviceIndex], RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(ordinal), &ordinal );
    return ordinal;
  }
  return -1;
}


void OceanScene::initScene( InitialCameraData& camera_data )
{
  try {

    //
    // Setup OptiX state
    //
    m_context->setRayTypeCount( 1 );
    m_context->setEntryPointCount( 4 );
    m_context->setStackSize(2000);

    m_context["radiance_ray_type"   ]->setUint( 0u );
    m_context["scene_epsilon"       ]->setFloat( 1.e-3f );
    m_context["max_depth"           ]->setInt( 6 );

    // Declare camera variables.  The values do not matter, they will be overwritten in trace.
    m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["U"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["V"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    m_context["W"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
    
    // Set up camera
    camera_data.eye    = make_float3( 1.47502f, 0.284192f, 0.8623f );
    camera_data.lookat = make_float3( 0.0f, 0.0f, 0.0f );
    camera_data.up     = make_float3( 0.0f, 1.0f, 0.0f );
    camera_data.vfov   = 45.0f;
    
    // Exception program
    std::string ptx_path = ptxpath( "ocean", "accum_camera.cu" );
    Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
    m_context->setExceptionProgram( 0, exception_program );
    m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );

    // Ray gen program for raytracing camera
    Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
    m_context->setRayGenerationProgram( 0, ray_gen_program );
    m_output_buffer = createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT );
    m_context["output_buffer"]->set( m_output_buffer ); 
    m_accum_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, WIDTH, HEIGHT );
    m_context["accum_buffer"]->set( m_accum_buffer ); 
    m_context["pre_image"]->set( m_accum_buffer ); 
    m_context["frame"]->setInt( 1 ); 

    // Preetham sky model
    ptx_path = ptxpath("ocean", "ocean_render.cu" );
    m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
    m_context["cutoff_color" ]->setFloat( 0.07f, 0.18f, 0.3f );

    // RNG seed buffer
    genRndSeeds( WIDTH, HEIGHT );

    // Ray gen program for heightfield update
    ptx_path = ptxpath( "ocean", "ocean_sim.cu" );
    Program data_gen_program = m_context->createProgramFromPTXFile( ptx_path, "generate_spectrum" );
    m_context->setRayGenerationProgram( 1, data_gen_program );
    m_context["patch_size"]->setFloat( m_patch_size );
    m_context["t"]->setFloat( 0.0f );
    m_h0_buffer = m_context->createBuffer( RT_BUFFER_INPUT,  RT_FORMAT_FLOAT2, FFT_WIDTH, FFT_HEIGHT );
    m_ht_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT2, FFT_WIDTH, FFT_HEIGHT ); 
    m_ik_ht_buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT2, FFT_WIDTH, FFT_HEIGHT ); 
    m_context["h0"]->set( m_h0_buffer ); 
    m_context["ht"]->set( m_ht_buffer ); 
    m_context["ik_ht"]->set( m_ik_ht_buffer ); 
    
    //Ray gen program for normal calculation
    Program normal_program = m_context->createProgramFromPTXFile( ptx_path, "calculate_normals" );
    m_context->setRayGenerationProgram( 2, normal_program );
    m_context["height_scale"]->setFloat( 0.5f );
    // Could pack data and normals together, but that would preclude using fft_output directly as data_buffer.
    m_data_buffer   = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT,
                                             HEIGHTFIELD_WIDTH,
                                             HEIGHTFIELD_HEIGHT );
    m_normal_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4,
                                             HEIGHTFIELD_WIDTH,
                                             HEIGHTFIELD_HEIGHT );
    m_context["data"]->set(m_data_buffer);
    m_context["normals"]->set(m_normal_buffer );

    // Ray gen program for tonemap
    ptx_path = ptxpath( "ocean", "tonemap.cu" );
    Program tonemap_program = m_context->createProgramFromPTXFile( ptx_path, "tonemap" );
    m_context->setRayGenerationProgram( 3, tonemap_program );
    m_context["f_exposure"]->setFloat( 0.0f );


    // Set up light buffer
    m_context["ambient_light_color"]->setFloat( 1.0f, 1.0f, 1.0f );
    BasicLight lights[] = { 
      { { 4.0f, 12.0f, 10.0f }, { 1.0f, 1.0f, 1.0f }, 1 }
    };

    Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(BasicLight));
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    m_context["lights"]->set(light_buffer);

    createGeometry();

    //
    // Sun and sky model
    //
    m_sun_sky.setSunTheta( 1.2f );
    m_sun_sky.setSunPhi( 0.0f );
    m_sun_sky.setTurbidity( 2.2f );
    m_sun_sky.setVariables( m_context );


    //
    // Setup cufft state
    //

    const unsigned int fft_input_size  = FFT_WIDTH * FFT_HEIGHT * sizeof(float2);
    const unsigned int fft_output_size = HEIGHTFIELD_WIDTH * HEIGHTFIELD_HEIGHT * sizeof(float);

    m_context->launch( 0, 0 );

    int firstOrdinal = GetOptixDeviceOrdinal( m_context, 0 );

    if ( firstOrdinal >= 0 )
    {
      // output the CUFFT results directly into Optix buffer
      cudaSetDevice(firstOrdinal);

      cutilSafeCall( cudaMalloc( reinterpret_cast<void**>( &m_d_h0 ), fft_input_size ) );
      cutilSafeCall( cudaMalloc( reinterpret_cast<void**>( &m_d_ht ), fft_input_size ) );
      cutilSafeCall( cudaMalloc( reinterpret_cast<void**>( &m_d_ht_real ), fft_output_size ) );
    }

    m_h_h0      = new float2[FFT_WIDTH * FFT_HEIGHT];
    m_h_ht      = new float2[FFT_WIDTH * FFT_HEIGHT];
    m_h_ht_real = new float [HEIGHTFIELD_WIDTH * HEIGHTFIELD_HEIGHT];
    generateH0( m_h_h0 );

    cutilSafeCall( cudaMemcpy( m_d_h0, m_h_h0, fft_input_size, cudaMemcpyHostToDevice) ); 

    memcpy( m_h0_buffer->map(), m_h_h0, fft_input_size ); 
    m_h0_buffer->unmap();

    // Finalize
    m_context->validate();
    m_context->compile();


  } catch( Exception& e ) {
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  } catch( ... ) {
    std::cerr << "UNKNOWN FAILURE ... exiting" << std::endl;
    exit(1);
  }
}


Buffer OceanScene::getOutputBuffer()
{
  return m_output_buffer;
}


void OceanScene::trace( const RayGenCameraData& camera_data )
{
  if( m_camera_changed || m_animate ) {
    m_camera_changed = false;
    m_frame = 0;
  }
  if( m_animate ) {
    updateHeightfield();
  }


  m_context["frame"]->setInt( m_frame );

  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  RTsize buffer_width, buffer_height;
  m_output_buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, 
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height)
                   );

  if( m_tonemap )
  m_context->launch( 3, 
                   static_cast<unsigned int>(buffer_width),
                   static_cast<unsigned int>(buffer_height)
                   );
  ++m_frame;
}

void OceanScene::createGeometry()
{
  Geometry heightfield = m_context->createGeometry();
  heightfield->setPrimitiveCount( 1u );
 
  
  heightfield["data"]->set(m_data_buffer );

  std::string ptx_path = ptxpath( "ocean", "ocean_render.cu" );
  heightfield->setBoundingBoxProgram(  m_context->createProgramFromPTXFile( ptx_path, "bounds" ) );
  heightfield->setIntersectionProgram( m_context->createProgramFromPTXFile( ptx_path, "intersect" ) );
  float3 min = make_float3( -2.0f, -0.2f, -2.0f );
  float3 max = make_float3(  2.0f,  0.2f,  2.0f );
  RTsize nx, nz;
  m_data_buffer->getSize(nx, nz);
  
  // If buffer is nx by nz, we have nx-1 by nz-1 cells;
  float3 cellsize = (max - min) / (make_float3(static_cast<float>(nx-1), 1.0f, static_cast<float>(nz-1)));
  cellsize.y = 1;
  float3 inv_cellsize = make_float3(1)/cellsize;
  heightfield["boxmin"]->setFloat(min);
  heightfield["boxmax"]->setFloat(max);
  heightfield["cellsize"]->setFloat(cellsize);
  heightfield["inv_cellsize"]->setFloat(inv_cellsize);

  // Create material
  Material heightfield_matl = m_context->createMaterial();
  Program water_ch = m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" );

  heightfield_matl["importance_cutoff"  ]->setFloat( 0.01f );
  //heightfield_matl["cutoff_color"       ]->setFloat( 0.27f, 0.48f, 0.6f );
  heightfield_matl["fresnel_exponent"   ]->setFloat( 4.0f );
  heightfield_matl["fresnel_minimum"    ]->setFloat( 0.05f );
  heightfield_matl["fresnel_maximum"    ]->setFloat( 0.30f );
  heightfield_matl["refraction_index"   ]->setFloat( 1.4f );
  heightfield_matl["refraction_color"   ]->setFloat( 0.95f, 0.95f, 0.95f );
  heightfield_matl["reflection_color"   ]->setFloat( 0.7f, 0.7f, 0.7f );
  heightfield_matl["refraction_maxdepth"]->setInt( 6 );
  heightfield_matl["reflection_maxdepth"]->setInt( 6 );
  float3 extinction = make_float3(.75f, .89f, .80f);
  heightfield_matl["extinction_constant"]->setFloat( log(extinction.x), log(extinction.y), log(extinction.z) );
  heightfield_matl["shadow_attenuation"]->setFloat( 1.0f, 1.0f, 1.0f );
  heightfield_matl->setClosestHitProgram( 0, water_ch);

  heightfield_matl["Ka"]->setFloat(0.0f, 0.3f, 0.1f);
  heightfield_matl["Kd"]->setFloat(0.3f, 0.7f, 0.5f);
  heightfield_matl["Ks"]->setFloat(0.1f, 0.1f, 0.1f);
  heightfield_matl["phong_exp"]->setFloat(1600);
  heightfield_matl["reflectivity"]->setFloat(0.1f, 0.1f, 0.1f);

  GeometryInstance gi = m_context->createGeometryInstance( heightfield, &heightfield_matl, &heightfield_matl+1 );
  
  GeometryGroup geometrygroup = m_context->createGeometryGroup();
  geometrygroup->setChildCount( 1 );
  geometrygroup->setChild( 0, gi );

  geometrygroup->setAcceleration( m_context->createAcceleration("NoAccel","NoAccel") );
  
  m_context["top_object"]->set( geometrygroup );
  m_context["top_shadower"]->set( geometrygroup );
}

void OceanScene::doResize( unsigned int width, unsigned int height )
{
  // output_buffer resizing handled in base class
  m_accum_buffer->setSize( width, height );
  m_seed_buffer->setSize( width, height );
  genRndSeeds( width, height );
}


void OceanScene::genRndSeeds( unsigned int width, unsigned int height )
{
  // Init random number buffer if necessary.
  if( m_seed_buffer.get() == 0 ) {
    m_seed_buffer  = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT,
                                              WIDTH, HEIGHT);
    m_context["rnd_seeds"]->setBuffer(m_seed_buffer);
  }

  unsigned int* seeds = static_cast<unsigned int*>( m_seed_buffer->map() );
  fillRandBuffer(seeds, width*height);
  m_seed_buffer->unmap();
}


void OceanScene::updateHeightfield()
{
  const unsigned int input_size  = FFT_WIDTH*FFT_HEIGHT*sizeof( float2 );

  double current_time;
  sutilCurrentTime( &current_time );
  if( m_last_time == 0.0 ) m_last_time = current_time;
  m_anim_time += current_time - m_last_time;
  m_last_time  = current_time;
  m_context["t"]->setFloat( static_cast<float>(m_anim_time) * -0.5f * m_anim_time_scale );

  m_context->launch( 1, FFT_WIDTH, FFT_HEIGHT );
  memcpy( m_h_ht, m_ht_buffer->map(), input_size );
  m_ht_buffer->unmap();

  int firstOrdinal = GetOptixDeviceOrdinal( m_context, 0 );

  if ( firstOrdinal >= 0 )
  {
    // output the CUFFT results directly into Optix buffer
    void* m_data_buffer_device_ptr = rtGetBDP<void*>( m_context, m_data_buffer->get(), 0 );
    cudaSetDevice(firstOrdinal);
    cutilSafeCall( cudaMemcpy( m_d_ht, m_h_ht, input_size, cudaMemcpyHostToDevice ) );
    cufftSafeCall( cufftPlan2d( &m_fft_plan, HEIGHTFIELD_WIDTH, HEIGHTFIELD_HEIGHT, CUFFT_C2R) );
    cufftSafeCall( cufftExecC2R( m_fft_plan, reinterpret_cast<cufftComplex*>( m_d_ht ),
          reinterpret_cast<cufftReal*>( m_data_buffer_device_ptr ) ) );
  }

  cufftSafeCall( cufftDestroy( m_fft_plan ) );

  m_context->launch( 2, HEIGHTFIELD_WIDTH, HEIGHTFIELD_HEIGHT );
}


// Phillips spectrum
// Vdir - wind angle in radians
// V - wind speed
float OceanScene::phillips(float Kx, float Ky, float Vdir, float V, float A)
{
  const float g = 9.81f;            // gravitational constant

  float k_squared = Kx * Kx + Ky * Ky;
  float k_x = Kx / sqrtf(k_squared);
  float k_y = Ky / sqrtf(k_squared);
  float L = V * V / g;
  float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

  if (k_squared == 0.0f ) return 0.0f;
  return A * expf( -1.0f / (k_squared * L * L) ) / (k_squared * k_squared) * w_dot_k * w_dot_k;
}


// Generate base heightfield in frequency space
void OceanScene::generateH0( float2* h_h0 )
{
  for (unsigned int y = 0u; y < FFT_HEIGHT; y++) {
    for (unsigned int x = 0u; x < FFT_WIDTH; x++) {
      float kx = M_PIf * x / m_patch_size;
      float ky = 2.0f * M_PIf * y / m_patch_size;

      // note - these random numbers should be from a Gaussian distribution really
      float Er = 2.0f * rand() / static_cast<float>( RAND_MAX ) - 1.0f;
      float Ei = 2.0f * rand() / static_cast<float>( RAND_MAX ) - 1.0f;

      // These can be made user-adjustable
      const float wave_scale = .00000000775f;
      const float wind_speed = 10.0f;     
      const float wind_dir   = M_PIf/3.0f;   

      float P = sqrtf( phillips( kx, ky, wind_dir, wind_speed, wave_scale ) );


      float h0_re = 1.0f / sqrtf(2.0f) * Er * P;
      float h0_im = 1.0f / sqrtf(2.0f) * Ei * P;

      int i = y*FFT_WIDTH+x;
      h_h0[i].x = h0_re;
      h_h0[i].y = h0_im;
      
      if( (x == 0  ) ) {
        h_h0[i].x = h_h0[i].y = 0.0f;
      }
      
    }
  }
}

//-----------------------------------------------------------------------------
//
// main 
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << std::endl;
  
  std::cerr
    << "App keystrokes:\n"
    << "  't' Toggle on/off tonemapping.\n"
    << "  ' ' Toggle on/off water animation.  Pixel AA enabled when animation is off.\n" 
    << std::endl;

  GLUTDisplay::printUsage();

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  printUsageAndExit( argv[0], false );

  try {
    OceanScene scene;
    GLUTDisplay::run( "OceanScene", &scene, GLUTDisplay::CDAnimated );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}

