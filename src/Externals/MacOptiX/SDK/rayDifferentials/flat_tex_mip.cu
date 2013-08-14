
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

using namespace optix;

struct PerRayData_radiance_differentials
{
  float3 result;
  float  importance;
  int    depth;
  
  float3 origin_dx;
  float3 direction_dx;
  float3 origin_dy;
  float3 direction_dy;
};

struct PerRayData_shadow
{
  float attenuation;
};

rtDeclareVariable(PerRayData_radiance_differentials, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, dudP, attribute dudP, ); 
rtDeclareVariable(float3, dvdP, attribute dvdP, ); 
rtBuffer<uchar4, 2> output_buffer;

rtDeclareVariable(int, num_mip_levels, , );
rtDeclareVariable(int2, tex0_dim, , );
rtDeclareVariable(float2, tex_scale, , );


rtBuffer<int, 1> tex_ids;

enum MIPMode {
  MIP_Disable = 0,
  MIP_Enable,
  MIP_FalseColor,

  NumMIPModes
};

rtDeclareVariable(int, mip_mode, , ) = MIP_Enable;

static __inline__ __device__ float4 get_color(int i)
{
  const float4 color_ramp[16] = {

    {1.0f,  0.0f,  0.0f,  1.f},
    {1.0f,  0.35f, 0.0f,  1.f},
    {1.0f,  0.75f, 0.0f,  1.f},
    {1.0f,  1.0f,  0.0f,  1.f},
    {0.5f,  1.0f,  0.0f,  1.f},
    {0.11f, 1.0f,  0.0f,  1.f},
    {0.0f,  1.0f,  0.26f, 1.f},
    {0.0f,  1.0f,  1.0f,  1.f},
    {0.0f,  0.61f, 1.0f,  1.f},
    {0.0f,  0.26f, 1.0f,  1.f},
    {0.0f,  0.0f,  1.0f,  1.f},
    {0.4f,  0.0f,  1.0f,  1.f},
    {0.6f,  0.0f,  1.0f,  1.f},
    {1.0f,  0.0f,  1.0f,  1.f},
    {1.0f,  0.2f,  1.0f,  1.f},
    {1.0f,  0.6f,  1.0f,  1.f}
  };
 
  if(mip_mode == MIP_FalseColor) {
    return color_ramp[i];
  } else {
    float2 uv = make_float2(texcoord)*tex_scale;
    return rtTex2D<float4>(tex_ids[i], uv.x, uv.y);
  }
}


RT_PROGRAM void closest_hit_radiance()
{
  size_t2 screen = output_buffer.size();
  float2 pixel_delta = 2.f / make_float2(screen);

  float3 dPdx = differential_transfer_origin(prd_radiance.origin_dx,
                                             prd_radiance.direction_dx,
                                             t_hit,
                                             ray.direction,
                                             shading_normal);
  
  float3 dPdy = differential_transfer_origin(prd_radiance.origin_dy,
                                             prd_radiance.direction_dy,
                                             t_hit,
                                             ray.direction,
                                             shading_normal);

  

  // texture space footprint
  float2 duvdx = make_float2( dot(dudP, dPdx), dot(dvdP, dPdx) );
  float2 duvdy = make_float2( dot(dudP, dPdy), dot(dvdP, dPdy) );

  float2 dstdu = make_float2( tex0_dim.x, 0.f )*tex_scale;
  float2 dstdv = make_float2( 0.f, tex0_dim.y )*tex_scale;

  float2 dstdx = dstdu * duvdx + dstdv * duvdx;
  float2 dstdy = dstdu * duvdy + dstdv * duvdy;

  float2 deltaTx = dstdx * pixel_delta.x;
  float2 deltaTy = dstdy * pixel_delta.y;

  float lod;
  if(mip_mode != MIP_Disable)
    lod = log2(max( length(deltaTx), length(deltaTy) ));
  else
    lod = 0;

  float4 color;
  if(lod >= num_mip_levels-1) {
    color = get_color(num_mip_levels-1);
  } else if( lod <= 0 ) {
    color = get_color(0);
  } else {
    int lod_lo = floorf(lod);
    int lod_hi = ceilf(lod);
    float t = lod - (float)((int)lod);
    color = (1.f-t)*get_color(lod_lo) + t*get_color(lod_hi);
  }

  prd_radiance.result = make_float3(color);
}

RT_PROGRAM void any_hit_shadow()
{
  prd_shadow.attenuation = 0.f;
  rtTerminateRay();
}
