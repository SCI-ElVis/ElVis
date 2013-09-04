
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "helpers.h"
#include "zoneplate_common.h"

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtBuffer<uchar4, 2>              output_color_only;
rtBuffer<unsigned char, 2>       adaptive_sample_locations;
rtBuffer<zpSample, 2>            output_samples;
rtBuffer<float, 2>               filter_weights;
rtBuffer<float, 2>               weighted_scatter_sums;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, window_size, ,);
rtDeclareVariable(int, filter_type, ,);
rtDeclareVariable(int, render_type, ,);
rtDeclareVariable(float, filter_width, ,);
rtDeclareVariable(float, checkerboard_rotate, ,);
rtDeclareVariable(int, checkerboard_width, ,);
rtDeclareVariable(uint, sqrt_samples_per_pixel, ,);
rtDeclareVariable(float, gaussian_alpha, ,);
rtDeclareVariable(float, jitter_amount, ,);
rtDeclareVariable(float, sinc_tau, ,);
rtDeclareVariable(uint, contrast_window_width, ,);
rtDeclareVariable(float, adaptive_contrast_threshold, ,);

static __device__ float zoneplate(float2 loc)
{
  uint2 size = window_size;
  float2 uv = loc / make_float2(size);
  float r = sqrtf(dot(uv,uv));
  return (1.0f+cos(2000.0f*r*r))/2.f;
}

static __device__ float checkerboard(float2 loc)
{
  int2 rotloc;
  float angle = checkerboard_rotate * M_PIf / 180.f;
  float ca = cosf(angle), sa = sinf(angle);
  rotloc.x = abs((int)floor(((ca*loc.x + sa * loc.y) / checkerboard_width)));
  rotloc.y = abs((int)floor(((ca*loc.y - sa * loc.x) / checkerboard_width)));
  
  if (rotloc.x % 2 == rotloc.y % 2) return 1.0f;
  return 0.0f;  
}

static __device__  float computeResult( float2 loc )
{
  if (render_type == RENDER_ZONEPLATE)
    return zoneplate(loc);
  else if (render_type == RENDER_CHECKERBOARD)
    return checkerboard(loc);
    
  return 0;
}

RT_PROGRAM void zp_color_only()
{
  float result = computeResult( make_float2(launch_index.x + 0.5f, launch_index.y + 0.5f) );
  output_color_only[launch_index] = make_color( make_float3(result) );
}

static __device__ float2 get_new_sample( uint2 corner )
{
  float2 loc = make_float2( (corner.x + 0.5f) / sqrt_samples_per_pixel,
                            (corner.y + 0.5f) / sqrt_samples_per_pixel ); 

  return loc;
}

RT_PROGRAM void zp_generate_samples()
{
  float2 loc = get_new_sample(launch_index);
  output_samples[launch_index].x = loc.x;
  output_samples[launch_index].y = loc.y;
  output_samples[launch_index].value = computeResult(loc);
}

static __device__ float mitchell1D( float x )
{
  float B = 1.f/3.f;
  float C = 1.f/3.f;
  
  if (x > 1.f)
    return ((-B - 6*C) * x*x*x + (6*B + 30*C) * x*x +
            (-12*B - 48*C) * x + (8*B + 24*C)) * (1.f/6.f);
  else
    return ((12 - 9*B - 6*C) * x*x*x +
            (-18 + 12*B + 6*C) * x*x +
            (6 - 2*B)) * (1.f/6.f);
}

static __device__ float sinc1D(float x) {
  if (x < 1e-5f) return 1.f;
  if (x > 1.f) return 0.f;
  x *= M_PIf;
  float sinc = sinf(x * sinc_tau) / (x * sinc_tau);
  float lanczos = sinf(x) / x;
  return sinc * lanczos;
}

// takes the sample and the pixel loc.
static __device__ float evaluate_filter( float2 sample, uint2 pi ) {
  // need the .5 because we want to consider, e.g., 
  //the (0,0) pixel to be (.5, .5) in continuous sample space.
  
  float dx = fabs(sample.x - (pi.x + .5f)); 
  float dy = fabs(sample.y - (pi.y + .5f));
  
  if (dx > filter_width || dy > filter_width ) return 0;
  
  float gaussian_exp;
    
  switch( filter_type ) {
    case FILTER_BOX:
      return 1.f;
    case FILTER_TRIANGLE:
      return max(0.f,filter_width - dx) * max(0.f,filter_width - dy);
    case FILTER_GAUSSIAN:
      gaussian_exp = expf(-gaussian_alpha * filter_width * filter_width);
      return max(0.f, expf(-gaussian_alpha*dx*dx) - gaussian_exp) * max(0.f, expf(-gaussian_alpha*dy*dy) - gaussian_exp);
    case FILTER_MITCHELL:
      return mitchell1D(dx/filter_width) * mitchell1D(dy/filter_width);
    case FILTER_SINC:
      return sinc1D(dx/filter_width) * sinc1D(dy/filter_width);
  }
  
  return 1; // should not happen
}

RT_PROGRAM void zp_gather_samples()
{
  // figure out the x,y extent of all samples that might affect this output pixel;
  uint2 ll, ur;
  
  ll.x = max((int)floorf(sqrt_samples_per_pixel * (launch_index.x + .5f - filter_width)),0);
  ur.x = min((int)ceilf(sqrt_samples_per_pixel * (launch_index.x + .5f + filter_width)), sqrt_samples_per_pixel * window_size.x-1);
  ll.y = max((int)floorf(sqrt_samples_per_pixel * (launch_index.y + .5f - filter_width)),0);
  
  float num = 0.f;
  float denom = 0.f;
  
  while (ur.x-- != ll.x) {
    ur.y = min((int)ceilf(sqrt_samples_per_pixel * (launch_index.y + .5f + filter_width)), sqrt_samples_per_pixel * window_size.y-1);

    while (ur.y-- != ll.y) {
      uint2 sample_index = make_uint2(ur.x,ur.y);
      float filt = evaluate_filter( make_float2(output_samples[sample_index].x,output_samples[sample_index].y), launch_index );
      num += filt * output_samples[sample_index].value;
      denom += filt;
    }
  }

  output_color_only[launch_index] = make_color(make_float3(num/denom));
}

RT_PROGRAM void zp_zero_scatter_buffers()
{
  weighted_scatter_sums[launch_index] = 0.0f;
  filter_weights[launch_index] = 0.0f;
}


static __device__ float zp_scatter_one( float2 loc )
{  
  float val = computeResult(loc);
  
  bool sync = (filter_width > 0.5f);
  uint2 pi;
  
  // very small or very large jitters cause floating point errors here, so don't try to be
  // clever about atomics if the sample is super close to the pixel boundary.
  bool too_close_to_pixel_boundary = (floorf(loc.x)==loc.x || floorf(loc.y)==loc.y);
  
  if (!(sync || too_close_to_pixel_boundary)) {
    // just do the floor here, since we KNOW it's one pixel
    
    pi = make_uint2(floorf(loc.x),floorf(loc.y));
    float filt = evaluate_filter( loc, pi );
    weighted_scatter_sums[pi] += val * filt;
    filter_weights[pi] += filt;
  } else {
    // find all pixels affected by this sample
 
    float dimageX = loc.x - 0.5f;
    float dimageY = loc.y - 0.5f;
    int x0 = (int) ceilf (dimageX - filter_width);
    int x1 = (int) floorf(dimageX + filter_width);
    int y0 = (int) ceilf (dimageY - filter_width);
    x0 = max(x0, 0);
    x1 = min(x1, window_size.x-1);
    y0 = max(y0, 0);
    
  while (x1-- != x0) {
    int y1 = (int) floorf(dimageY + filter_width);
    y1 = min(y1, window_size.y-1);
    while (y1-- != y0) {
        pi = make_uint2(x1,y1);
        float filt = evaluate_filter( loc, pi );
        atomicAdd( &(weighted_scatter_sums[pi]), val * filt );
        atomicAdd( &(filter_weights[pi]), filt );
      }
    }
  }
  
  return val;
}

RT_PROGRAM void zp_scatter_samples()
{
  for (int i = 0 ; i < sqrt_samples_per_pixel ; i++) {
    for (int j = 0 ; j < sqrt_samples_per_pixel ; j++) { 
      uint2 corner = make_uint2( launch_index.x * sqrt_samples_per_pixel + i, launch_index.y * sqrt_samples_per_pixel + j );
      float2 loc = get_new_sample( corner );
      float val = zp_scatter_one( loc );
    }
  }
}

RT_PROGRAM void zp_scatter_do_divide() {
  output_color_only[launch_index] = make_color(make_float3(weighted_scatter_sums[launch_index] / filter_weights[launch_index]));
}

RT_PROGRAM void zp_find_contrast_locations() {
   uint x0, x1, y0, y1;
  
  x0 = max( (int) (launch_index.x - contrast_window_width), 0 );
  y0 = max( (int) (launch_index.y - contrast_window_width), 0 );
  x1 = min( launch_index.x + contrast_window_width, window_size.x - 1);
  
  uint window_max = 0, window_min = 255;
  
  while (x1-- != x0)
  {
    y1 = min( launch_index.y + contrast_window_width, window_size.y - 1);
    while (y1-- != y0)
    {
      uint2 loc = make_uint2( x1,y1 );
      uint val = output_color_only[loc].x;
      window_max = max(window_max, val);
      window_min = min(window_min, val);
    }
  }
  
  float contrast;
  
  if (window_max == 0)
  {
    contrast = 0;
  }
  else
    contrast = float(window_max-window_min)/(window_max+window_min);
  
  adaptive_sample_locations[launch_index] = (contrast >= adaptive_contrast_threshold ? 255 : 0);
}

RT_PROGRAM void zp_coalesce_adaptive_sample_locations() {
}

RT_PROGRAM void zp_adaptive_resample() {
  if (adaptive_sample_locations[launch_index] == 0) {
    return; // nothing to do!
  }

  for (int i = 0 ; i < sqrt_samples_per_pixel ; i++) {
    for (int j = 0 ; j < sqrt_samples_per_pixel ; j++) { 
      uint2 corner = make_uint2( launch_index.x * sqrt_samples_per_pixel + i, launch_index.y * sqrt_samples_per_pixel + j );
      float2 loc = get_new_sample( corner );
      float val = zp_scatter_one( loc );
    }
  }
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
}
