
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

//------------------------------------------------------------------------------
//
// mcmc_sampler.cpp: render an OBJ file using a Markov chain Monte Carlo method
//
//------------------------------------------------------------------------------

#include <optixu/optixpp_namespace.h>
#include <sutil.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <GLUTDisplay.h>
#include <ObjLoader.h>
#include <string>
#include <SunSky.h>
#include "mcmc_sampler.h"

using namespace optix;

//-----------------------------------------------------------------------------
//
// Helpers
//
//-----------------------------------------------------------------------------

namespace
{
  std::string ptxpath( const std::string& base )
  {
    return std::string(sutilSamplesPtxDir()) + "/mcmc_sampler_generated_" + base + ".ptx";
  }
}

//-----------------------------------------------------------------------------
//
// MCMCSamplerScene
//
//-----------------------------------------------------------------------------

class MCMCSamplerScene: public SampleScene
{
public:
  // Set the actual render parameters below in main().
  MCMCSamplerScene( unsigned int m_scene_index )
  : m_sun_scale( 1.0f / 3200000.0f )
  , m_sky_scale( 4.0f / 100.0f )
  , sampler_type(SamplerRandomWalk)
  , m_width(512u)
  , m_height(512u)
  , m_num_chains(128u * 128u)
  , m_num_initial_passes(128u)
  , m_num_passes(16u)
  , m_initialization_count(m_num_initial_passes)
  , m_scene_index(m_scene_index)
  {
    init_mcs.resize(m_num_chains * m_num_initial_passes);
    init_mcsf.resize(m_num_chains * m_num_initial_passes);
    init_mcs_accum.resize(m_num_chains * m_num_initial_passes);
  }

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer();
  virtual void   doResize( unsigned int width, unsigned int height );
  bool   keyPressed( unsigned char key, int x, int y );

  void   setDimensions( const unsigned int w, const unsigned int h ) { m_width = w; m_height = h; }

  void   setSceneIndex( const unsigned int scene_index );

private:
  void createGeometry();

  std::string       m_filename[3];
  GeometryGroup     m_geometry_group[3];
  optix::Aabb       m_aabb[3];
  InitialCameraData m_camera[3];
  optix::Buffer     m_light_buffer[3];
  Material          m_material;


  std::vector<int2>  init_mcs;
  std::vector<float> init_mcsf;
  std::vector<float> init_mcs_accum;  

  PreethamSunSky m_sun_sky;
  double         m_sun_scale;
  double         m_sky_scale;

  ESamplerType   sampler_type;

  unsigned int   m_width;
  unsigned int   m_height;
  unsigned int   m_frame;
  unsigned int   m_num_chains;
  unsigned int   m_num_initial_passes;
  unsigned int   m_num_passes;

  unsigned int   m_initialization_count;

  unsigned int   m_scene_index;
};

// Return whether we processed the key or not
bool MCMCSamplerScene::keyPressed( unsigned char key, int x, int y )
{
  switch ( key )
  {
  case '1': {
    sampler_type = SamplerUniform;
    m_context["sampler_type"]->setUint(sampler_type);
    m_camera_changed = true;
    std::cerr << "Using uniform random sampler..." << std::endl;
    return true;
  } break;
  case '2': {
    sampler_type = SamplerRandomWalk;
    m_context["sampler_type"]->setUint(sampler_type);
    m_camera_changed = true;
    std::cerr << "Using random walk Markov chain Monte Carlo sampler..." << std::endl;
    return true;
  } break;
  case 'v': {
    m_scene_index = (m_scene_index + 1) % 3;
    m_context["top_object"]->set( m_geometry_group[ m_scene_index ] );
    m_context["light_buffer"]->set( m_light_buffer[ m_scene_index ] );
    GLUTDisplay::setCamera( m_camera[ m_scene_index ] );
    m_camera_changed = true;
    return true;
  } break;
  }

  return false;
}

void MCMCSamplerScene::initScene( InitialCameraData& camera_data )
{
  // Setup context

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

  m_context->setRayTypeCount( 2 ); 
  m_context->setEntryPointCount( 2 );
  m_context->setStackSize( 1040 );

  m_context["normalization_const"]->setFloat( 1.0f );
  m_context["scene_epsilon"]->setFloat( 1.e-2f );
  m_context["mcmctrace_ray_type"]->setUint(0u);
  m_context["mcmctrace_shadow_ray_type"]->setUint(1u);
  m_context["frame_width"]->setUint(m_width);
  m_context["frame_height"]->setUint(m_height);
  m_context["num_chains"]->setUint(m_num_chains);
  m_context["initial_sampling"]->setUint(1);
  m_context["sampler_type"]->setUint(sampler_type);

  // Setup output buffer
  Variable output_buffer = m_context["output_buffer"];
  Buffer buffer = createOutputBuffer( RT_FORMAT_FLOAT4, m_width, m_height );
  output_buffer->set(buffer);

  // Declare these so validation will pass
  m_context["eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["U"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["V"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context["W"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

  m_context["bad_color"]->setFloat( 0.0f, 1.0f, 0.0f );
  m_context["bg_color"]->setFloat( make_float3(0.0f) );

  // Seed buffer
  Buffer init_seed_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, m_num_chains );
  m_context["init_seed_buffer"]->set( init_seed_buffer );
  
  Buffer seed_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT2, m_num_chains );
  m_context["seed_buffer"]->set( seed_buffer );
  {
    Buffer buffer = m_context["seed_buffer"]->getBuffer();
    unsigned int* seeds = reinterpret_cast<unsigned int*>( buffer->map() );
    
    for (unsigned int i = 0; i < static_cast<unsigned int>(m_num_chains * 2); i++)
    {
      seeds[i] = i;
    }
    
    buffer->unmap();
  }

  // State buffer
  Buffer state_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT );
  state_buffer->setFormat( RT_FORMAT_USER );
  state_buffer->setElementSize( sizeof(MarkovChainState) );
  state_buffer->setSize( m_num_chains );
  m_context["state_buffer"]->set( state_buffer );
  {
    Buffer buffer = m_context["state_buffer"]->getBuffer();
    MarkovChainState* rnds = reinterpret_cast<MarkovChainState*>( buffer->map() );
    for( unsigned int i=0; i< m_num_chains; ++i )
    { 
      for( unsigned int j=0; j < RNDS_COUNT; ++j )
      {
        rnds[i].rnds[j] = 0.0f;
      }
      rnds[i].f = 0.0f;
      rnds[i].c = make_float3(0.0f);
    }
    buffer->unmap();
  }

  // Accumulation buffer
  Buffer accumulation_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT|RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, m_width, m_height );
  m_context["accumulation_buffer"]->set( accumulation_buffer );
  {
    Buffer buffer = m_context["accumulation_buffer"]->getBuffer();
    float* accumulations = reinterpret_cast<float*>( buffer->map() );
    for (unsigned int i = 0; i < static_cast<unsigned int>(m_width * m_height * 4); i ++)
    {
      accumulations[i] = 0.0f;
    }
    buffer->unmap();
  }

  // Setup programs
  std::string ptx_path = ptxpath( "mcmc_sampler", "mcmc_sampler.cu" );
  Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "trace_camera" );
  m_context->setRayGenerationProgram( 0, ray_gen_program );
  Program exception_program = m_context->createProgramFromPTXFile( ptx_path, "exception" );
  m_context->setExceptionProgram( 0, exception_program );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );

  std::string ptx_path3 = ptxpath( "mcmc_sampler", "mcmc_draw_result.cu" );
  Program ray_gen_program3 = m_context->createProgramFromPTXFile( ptx_path3, "draw_result" );
  m_context->setRayGenerationProgram( 1, ray_gen_program3 );

  m_material = m_context->createMaterial();
  Program m_material_ch = m_context->createProgramFromPTXFile( ptxpath( "mcmc_sampler", "mcmc_sampler.cu" ), "uber_material" );
  Program m_material_ah = m_context->createProgramFromPTXFile( ptxpath( "mcmc_sampler", "mcmc_sampler.cu" ), "shadow" );
  m_material->setClosestHitProgram( 0, m_material_ch );
  m_material->setAnyHitProgram( 1, m_material_ah );

  m_sun_sky.setSunTheta( 0.2f );
  m_sun_sky.setSunPhi( 2.85f );
  m_sun_sky.setTurbidity( 2.5f );
  m_sun_sky.setVariables( m_context );

  m_context["sun_scale"]->setFloat((float)m_sun_scale);
  m_context["sky_scale"]->setFloat((float)m_sky_scale);

  m_context["directional_light"]->setFloat(m_sun_sky.getSunDir());
  m_context["directional_light_col"]->setFloat(m_sun_sky.sunColor() * (float)m_sun_scale);
 
  m_context["frame_number"]->setUint(1);

  // Create scene geometry
  m_filename[0] = std::string( sutilSamplesDir() ) + "/mcmc_sampler/data/pool.obj";
  m_filename[1] = std::string( sutilSamplesDir() ) + "/mcmc_sampler/data/cornell.obj";
  m_filename[2] = std::string( sutilSamplesDir() ) + "/mcmc_sampler/data/ring.obj";

  createGeometry();

  // Set up cameras
  m_camera[0] = InitialCameraData( make_float3(24.5f, 2005.8f, 1019.1f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.45f, -0.89f), 30.0f );

  float max_dim = fmaxf( m_aabb[1].extent( 0 ), m_aabb[1].extent( 1 ) ); 
  float3 eye = m_aabb[1].center();
  eye.z += 2.25f * max_dim;
  m_camera[1] = InitialCameraData( eye, m_aabb[1].center(), make_float3( 0.0f, 1.0f, 0.0f ), 30.0f );  

  m_camera[2] = InitialCameraData( make_float3(325.0f, 639.0f, 804.9f), make_float3(295.9f, 173.6f, 398.9f), make_float3(0.038f, 0.656f, -0.753f), 30.0f );

  // set the default scene
  camera_data = m_camera[m_scene_index];  
  m_context["top_object"]->set( m_geometry_group[m_scene_index] );
  m_context["light_buffer"]->set( m_light_buffer[m_scene_index] );

  // Finalize
  m_context->validate();
  m_context->compile();
}


void MCMCSamplerScene::doResize(unsigned int width, unsigned int height)
{
  m_context["accumulation_buffer"]->getBuffer()->setSize( width, height );
  m_width = width;
  m_height = height;
  m_context["frame_width"]->setUint(m_width);
  m_context["frame_height"]->setUint(m_height);

  // Accumulation buffer
  {
    Buffer buffer = m_context["accumulation_buffer"]->getBuffer();
    float* accumulations = reinterpret_cast<float*>( buffer->map() );
    for (unsigned int i = 0; i < static_cast<unsigned int>(m_width * m_height * 4); i ++)
    {
      accumulations[i] = 0.0f;
    }
    buffer->unmap();
  }

  m_frame = 0;
  
  if (sampler_type == SamplerUniform)
  {
    m_frame = 1;
  }

  m_initialization_count = m_num_initial_passes;  
}


void MCMCSamplerScene::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat( camera_data.eye );
  m_context["U"]->setFloat( camera_data.U );
  m_context["V"]->setFloat( camera_data.V );
  m_context["W"]->setFloat( camera_data.W );

  if( m_camera_changed ) {
    m_camera_changed = false;
    m_frame = 0;

    {
      Buffer buffer = m_context["accumulation_buffer"]->getBuffer();
      float* accumulations = reinterpret_cast<float*>( buffer->map() );
      for (unsigned int i = 0; i < static_cast<unsigned int>(m_width * m_height * 4); i ++)
      {
        accumulations[i] = 0.0f;
      }
      buffer->unmap();
    }    
    
    if (sampler_type == SamplerUniform)
    {
      m_frame = 1;
    }
    
    m_initialization_count = m_num_initial_passes;
  }

  if (sampler_type != SamplerUniform)
  {
    if (m_initialization_count > 0)
    {
      //-----------------------------------------------------------------------------
      // initialization
      //-----------------------------------------------------------------------------
      // clear the accumulation buffer
      {
        Buffer buffer = m_context["accumulation_buffer"]->getBuffer();
        float* accumulations = reinterpret_cast<float*>( buffer->map() );
        for (unsigned int i = 0; i < static_cast<unsigned int>(m_width * m_height * 4); i ++)
        {
          accumulations[i] = 0.0f;
        }
        buffer->unmap();
      }  
      
      for (unsigned int n = 0; n < m_num_passes; n++)
      {
        m_context["frame_number"]->setUint( 0 );
        m_context["initial_sampling"]->setUint(1);
        m_context["normalization_const"]->setFloat( ((m_width * m_height) / float(m_num_chains)) / float(m_num_passes) );

        // sample paths
        unsigned int k = m_num_initial_passes - m_initialization_count;
        {
          // copy into the host memory
          {
            Buffer buffer = m_context["seed_buffer"]->getBuffer();
            int2* rnds = reinterpret_cast<int2*>( buffer->map() );
            for( unsigned int i=0; i < m_num_chains; ++i )
            { 
              init_mcs[i + k * m_num_chains] = rnds[i];
            }
            buffer->unmap();
          }
        
          // sample a batch of paths
          m_context->launch( 0, static_cast<unsigned int>(m_num_chains) );
          
          // copy into the host memory
          {
            Buffer buffer = m_context["state_buffer"]->getBuffer();
            MarkovChainState* rnds = reinterpret_cast<MarkovChainState*>( buffer->map() );
            for( unsigned int i=0; i < m_num_chains; ++i )
            { 
              init_mcsf[i + k * m_num_chains] = rnds[i].f;
            }
            buffer->unmap();
          }
        }
        m_initialization_count--;
        if (m_initialization_count == 0) break;
      }
    
      // initial sampling is done
      if (m_initialization_count == 0)
      {
        // compute the CDF
        init_mcs_accum[0] = init_mcsf[0];
        for( unsigned int i=1; i < m_num_chains * m_num_initial_passes; ++i )
        { 
          init_mcs_accum[i] = init_mcsf[i] + init_mcs_accum[i - 1];
        }
        
        // resampling
        {
          unsigned int j = 0;
          Buffer buffer = m_context["init_seed_buffer"]->getBuffer();
          int2* rnds = reinterpret_cast<int2*>( buffer->map() );
          float s = rand() / float(RAND_MAX);
          for( unsigned int i=0; i < m_num_chains; ++i )
          { 
            // stratified sampling on the CDF
            float value = (float(i) + s) / float(m_num_chains) * init_mcs_accum[m_num_chains * m_num_initial_passes - 1];
            while ( (init_mcs_accum[j] < value) && (j < (m_num_chains * m_num_initial_passes - 1)) )
            {
              j++;
            }
            rnds[i] = init_mcs[j];
          }
          buffer->unmap();
        }

        // set the normalization constant
        float b = init_mcs_accum[m_num_chains * m_num_initial_passes - 1] / float(m_num_chains  * m_num_initial_passes);
        m_context["normalization_const"]->setFloat( b * ((m_width * m_height) / float(m_num_chains)) );
        
        // clear the accumulation buffer
        {
          Buffer buffer = m_context["accumulation_buffer"]->getBuffer();
          float* accumulations = reinterpret_cast<float*>( buffer->map() );
          for (unsigned int i = 0; i < static_cast<unsigned int>(m_width * m_height * 4); i ++)
          {
            accumulations[i] = 0.0f;
          }
          buffer->unmap();
        }

        m_frame++;
      }

      m_context["frame_number"]->setUint( 1 );
    }
    else
    {
      for (unsigned int n = 0; n < m_num_passes; n++)
      {
        // MCMC sampler
        m_context["initial_sampling"]->setUint(0);
        m_context["frame_number"]->setUint( m_frame++ );  
        m_context->launch( 0, static_cast<unsigned int>(m_num_chains) );
      }
    }
  }
  else
  {
    for (unsigned int n = 0; n < m_num_passes; n++)
    {
      // uniform sampler
      m_context["normalization_const"]->setFloat( ((m_width * m_height) / float(m_num_chains)) );
      m_context["frame_number"]->setUint( m_frame++ );
      m_context->launch( 0, static_cast<unsigned int>(m_num_chains) );
    }
  }

  // draw the result
  m_context->launch( 1, static_cast<unsigned int>(m_width), static_cast<unsigned int>(m_height) );  
}

Buffer MCMCSamplerScene::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void MCMCSamplerScene::createGeometry()
{
  // Load OBJ scenes 
  for (unsigned int i = 0; i <= 2; i++)
  {
    m_geometry_group[i] = m_context->createGeometryGroup();

    ObjLoader* loader = 0;
    loader = new ObjLoader( m_filename[i].c_str(), m_context, m_geometry_group[i], m_material, true );
    loader->load();
    m_aabb[i] = loader->getSceneBBox();
    m_light_buffer[i] = loader->getLightBuffer();
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
    << "  -t  | --timeout <sec>                      Seconds before stopping rendering. Set to 0 for no stopping.\n"
    << "  -v  | --scene <N>                          Specify scene 0, 1, or 2. default=0.\n"
    << std::endl;

  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  1 Use uniform random sampler\n"
    << "  2 Use random walk Markov chain Monte Carlo sampler\n"
    << "  v Cycle scenes\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  unsigned int width = 512u, height = 512u;
  int scene_index = 0;
  float timeout = 0.0f;

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--timeout" || arg == "-t" ) {
      if(++i < argc) {
        timeout = static_cast<float>(atof(argv[i]));
      } else {
        std::cerr << "Missing argument to "<<arg<<"\n";
        printUsageAndExit(argv[0]);
      }
    } else if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if( arg == "-v" || arg == "--scene" ) {
      if(++i < argc) {
        scene_index = static_cast<unsigned int>(atoi(argv[i]));
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
    MCMCSamplerScene scene( scene_index );
    scene.setDimensions( width, height );
    GLUTDisplay::setProgressiveDrawingTimeout(timeout);
    GLUTDisplay::setUseSRGB(true);
    GLUTDisplay::run( "MCMC sampler", &scene, GLUTDisplay::CDProgressive );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
