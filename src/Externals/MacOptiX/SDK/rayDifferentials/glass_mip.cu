
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

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(int,          max_depth, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, shadow_ray_type, , );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, dNdP, attribute dNdP, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_dist, rtIntersectionDistance, );

rtDeclareVariable(float,        importance_cutoff, , );
rtDeclareVariable(float3,       cutoff_color, , );
rtDeclareVariable(float,        fresnel_exponent, , );
rtDeclareVariable(float,        fresnel_minimum, , );
rtDeclareVariable(float,        fresnel_maximum, , );
rtDeclareVariable(float,        refraction_index, , );
rtDeclareVariable(int,          refraction_maxdepth, , );
rtDeclareVariable(int,          reflection_maxdepth, , );
rtDeclareVariable(float3,       refraction_color, , );
rtDeclareVariable(float3,       reflection_color, , );

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
  float3 attenuation;
};

rtDeclareVariable(PerRayData_radiance_differentials, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// -----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_radiance()
{
  // intersection vectors
  const float3 h = ray.origin + isect_dist * ray.direction;            // hitpoint
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal
  const float3 i = ray.direction;                                            // incident direction
        float3 t;                                                            // transmission direction
        float3 r;                                                            // reflection direction

  float reflection = 1.0f;
  float3 result = make_float3(0.0f);
  
  const float depth = prd_radiance.depth;

  // refraction
  if (depth < min(refraction_maxdepth, max_depth))
  {
    if ( refract(t, i, n, refraction_index) )
    {

      // check for external or internal reflection
      float cos_theta = dot(i, n);
      if (cos_theta < 0.0f)
        cos_theta = -cos_theta;
      else
        cos_theta = dot(t, n);

      reflection = fresnel_schlick(cos_theta, fresnel_exponent, fresnel_minimum, fresnel_maximum);

      float importance = prd_radiance.importance * (1.0f-reflection) * optix::luminance( refraction_color );
      float3 color = cutoff_color;
      if ( importance > importance_cutoff ) {
        optix::Ray ray = optix::make_Ray( h, t, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
        PerRayData_radiance_differentials refr_prd;
        refr_prd.depth = depth+1;
        refr_prd.importance = importance;
        
        // transfer differentials
        refr_prd.origin_dx = differential_transfer_origin(prd_radiance.origin_dx, prd_radiance.direction_dx, isect_dist, ray.direction, shading_normal);
        refr_prd.origin_dy = differential_transfer_origin(prd_radiance.origin_dy, prd_radiance.direction_dy, isect_dist, ray.direction, shading_normal);
        refr_prd.direction_dx = prd_radiance.direction_dx;
        refr_prd.direction_dy = prd_radiance.direction_dy;

        // refract differentials
        refr_prd.origin_dx = refr_prd.origin_dx;
        refr_prd.origin_dy = refr_prd.origin_dy;
        refr_prd.direction_dx = differential_refract_direction(refr_prd.origin_dx, refr_prd.direction_dx, dNdP, i, n, refraction_index, t);
        refr_prd.direction_dy = differential_refract_direction(refr_prd.origin_dy, refr_prd.direction_dy, dNdP, i, n, refraction_index, t);
        
        rtTrace( top_object, ray, refr_prd );
        color = refr_prd.result;
      }
      result += (1.0f - reflection) * refraction_color * color;
    }
    // else TIR
  }

  // reflection
  if (depth < min(reflection_maxdepth, max_depth))
  {
    r = reflect(i, n);
  
    float importance = prd_radiance.importance * reflection * optix::luminance( reflection_color );
    float3 color = cutoff_color;
    if ( importance > importance_cutoff ) {
      optix::Ray ray = optix::make_Ray( h, r, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX );
      PerRayData_radiance_differentials refl_prd;
      refl_prd.depth = depth+1;
      refl_prd.importance = importance;

      // transfer differentials
      refl_prd.origin_dx = differential_transfer_origin(prd_radiance.origin_dx, prd_radiance.direction_dx, isect_dist, ray.direction, shading_normal);
      refl_prd.origin_dy = differential_transfer_origin(prd_radiance.origin_dy, prd_radiance.direction_dy, isect_dist, ray.direction, shading_normal);
      refl_prd.direction_dx = prd_radiance.direction_dx;
      refl_prd.direction_dy = prd_radiance.direction_dy;

      // reflect differentials
      refl_prd.origin_dx = refl_prd.origin_dx;
      refl_prd.origin_dy = refl_prd.origin_dy;
      refl_prd.direction_dx = differential_reflect_direction(refl_prd.origin_dx, refl_prd.direction_dx, dNdP, i, n);
      refl_prd.direction_dy = differential_reflect_direction(refl_prd.origin_dy, refl_prd.direction_dy, dNdP, i, n);
      
      rtTrace( top_object, ray, refl_prd );
      color = refl_prd.result;
    }
    result += reflection * reflection_color * color;
  }

  prd_radiance.result = result;
}

// -----------------------------------------------------------------------------

RT_PROGRAM void any_hit_shadow()
{
  prd_shadow.attenuation = make_float3(0.0f);

  rtTerminateRay();
}
