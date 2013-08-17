
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "random.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtBuffer<float4, 2>              variance_sum_buffer;
rtBuffer<float4, 2>              variance_sum2_buffer;
rtBuffer<unsigned int, 2>        num_samples_buffer;
rtBuffer<unsigned int, 2>        rnd_seeds;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );


// Check whether the pixel is in the center of block
static __device__ __inline__ bool shouldTrace( const uint2& index, unsigned int spacing )
{
  unsigned int half_spacing  = spacing >> 1;
  uint2        shifted_index = make_uint2( index.x + half_spacing, index.y + half_spacing ); 
  size_t2      screen        = output_buffer.size(); 
  return ( shifted_index.x % spacing == 0 && shifted_index.y % spacing == 0 ) ||
         ( index.x == screen.x-1 && screen.x % spacing <= half_spacing && shifted_index.y % spacing == 0 ) ||
         ( index.y == screen.y-1 && screen.y % spacing <= half_spacing && shifted_index.x % spacing == 0 );
}


// Flood fill a block
static __device__ __inline__ void fill( const uint2& index, const float3& color, unsigned int spacing )
{
  size_t2      screen        = output_buffer.size(); 
  unsigned int half_spacing  = spacing >> 1;

  unsigned int min_x = max( index.x-half_spacing, 0u );
  unsigned int max_x = min( index.x+half_spacing, (unsigned int) screen.x );
  unsigned int min_y = max( index.y-half_spacing, 0u );
  unsigned int max_y = min( index.y+half_spacing, (unsigned int) screen.y );
  
  for ( unsigned int i = min_x; i < max_x; ++i ) {
    for ( unsigned int j = min_y; j < max_y; ++j ) {
      output_buffer[ make_uint2( i, j ) ] = make_color(color);
    }
  }
}


// Trace ray through screen_coord
static __device__ __inline__ float3 trace( float2 screen_coord )
{
  size_t2 screen = output_buffer.size();
  float2 d = screen_coord / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);
  return prd.result;
}


// Will trace this pixel only if it is the center of a block of size block_size.  
static __device__ __inline__ void coarseTrace( const uint2& index, unsigned int block_size )
{
  if ( shouldTrace( index, block_size ) ) {
    float3 result = trace( make_float2( index ) ); 
    fill( index, result, block_size );
  }
}

static __device__ __inline__ float3 jittered_trace( const uint2& index )
{
    // Trace a randomly offset ray within the pixel
    volatile unsigned int seed  = rnd_seeds[ index ]; // volatile workaround for cuda 2.0 bug
    unsigned int new_seed  = seed;
    float uu = rnd( new_seed )-0.5f;
    float vv = rnd( new_seed )-0.5f;
    rnd_seeds[ launch_index ] = new_seed;

    float2 offset = make_float2( uu, vv );
    float3 result = trace( offset + make_float2( index ) );

    return result;
}

RT_PROGRAM void pinhole_camera()
{
  if      ( frame_number == 0 ) coarseTrace( launch_index, 8u );
  else if ( frame_number == 1 ) coarseTrace( launch_index, 4u );
  else if ( frame_number == 2 ) coarseTrace( launch_index, 2u );
  else if ( frame_number == 3 ) {

    float3 result = jittered_trace( launch_index );
    output_buffer[ launch_index ] = make_color( result );

    // Update buffers
    num_samples_buffer[ launch_index ]   = 1u;
    variance_sum_buffer[ launch_index ]  = make_float4(result, 0.0f);
    variance_sum2_buffer[ launch_index ] = make_float4(result*result, 0.0f);
  } else {
    {
      // ns < 0x80000000 means the variance is too high and we should keep rendering.
      volatile unsigned int ns = num_samples_buffer[ launch_index ];
      if ( (ns & 0x80000000) && (((launch_index.y >> 3) & 0x3) != (frame_number & 0x3)) ) {
        return;
      }
    }

    float3 new_color = jittered_trace( launch_index );
  
    // Add in new ray's contribution
    volatile unsigned int ns = num_samples_buffer[ launch_index ] & ~0x80000000; // volatile workaround for Cuda 2.0 bug
    float  new_value_weight = 1.0f / (float)ns;
    float  old_value_weight = 1.0f - new_value_weight;
    uchar4& old_bytes = output_buffer[ launch_index ];
    float3 old_color = make_float3(old_bytes.z, old_bytes.y, old_bytes.x)*make_float3(1.f/255.0f);
    float3 result = old_color*old_value_weight + new_color*new_value_weight;

    // Update buffers
    output_buffer[ launch_index ] = make_color(result); 
    float4 vsum  = variance_sum_buffer[ launch_index ];
    float4 vsum2 = variance_sum2_buffer[ launch_index ];
    // Compute the variance of the series of displayed pixels over time. This variance will go to zero, regardless of the variance of the sample values.
    variance_sum_buffer[ launch_index ]  = vsum  = vsum  + make_float4( result, 0.0f );
    variance_sum2_buffer[ launch_index ] = vsum2 = vsum2 + make_float4( result*result, 0.0f );
    ns++;

    // If we are beyond our first four samples per pixel, check variance
    if ( frame_number > 6 ) {
      float3 rgb_variance = ( make_float3( vsum2 ) - make_float3( vsum ) * make_float3( vsum ) * new_value_weight ) * new_value_weight;
      
      float variance = optix::luminance( rgb_variance ); 
      // render an 8-row span every 32 rows regardless. This shape lets entire warps turn off.
      if ( variance < 0.001f ) {
        ns = ns | 0x80000000;
      }
    }

    num_samples_buffer[ launch_index ] = ns;
  }
}


RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color(bad_color);
}
