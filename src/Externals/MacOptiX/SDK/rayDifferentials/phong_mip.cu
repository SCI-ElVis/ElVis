
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
#include "commonStructs.h"
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
  float3 attenuation;
};


rtDeclareVariable(int,               max_depth, , );
rtBuffer<BasicLight>                 lights;
rtDeclareVariable(float3,            ambient_light_color, , );
rtDeclareVariable(unsigned int,      radiance_ray_type, , );
rtDeclareVariable(unsigned int,      shadow_ray_type, , );
rtDeclareVariable(float,             scene_epsilon, , );
rtDeclareVariable(rtObject,          top_object, , );
rtDeclareVariable(rtObject,          top_shadower, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance_differentials, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

rtDeclareVariable(float3,       Ka, , );
rtDeclareVariable(float3,       Kd, , );
rtDeclareVariable(float3,       Ks, , );
rtDeclareVariable(float3,       reflectivity, , );
rtDeclareVariable(float,        phong_exp, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(float3, dNdP, attribute dNdP, ); 

static __device__ void phongShadowed()
{
  // this material is opaque, so it fully attenuates all shadow rays
  prd_shadow.attenuation = make_float3(0);
  
  rtTerminateRay();
}

static
__device__ void phongShade( float3 p_Kd,
                            float3 p_Ka,
                            float3 p_Ks,
                            float3 p_normal,
                            float  p_phong_exp,
                            float3 p_reflectivity )
{
  float3 hit_point = ray.origin + t_hit * ray.direction;
  
  // ambient contribution

  float3 result = p_Ka * ambient_light_color;

  // compute direct lighting
  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    BasicLight light = lights[i];
    float Ldist = length(light.pos - hit_point);
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot( p_normal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>( nDl > 0.0f ));
    if ( nDl > 0.0f && light.casts_shadow ) {
      PerRayData_shadow shadow_prd;
      shadow_prd.attenuation = make_float3(1.0f);
      optix::Ray shadow_ray( hit_point, L, shadow_ray_type, scene_epsilon, Ldist );
      rtTrace(top_shadower, shadow_ray, shadow_prd);
      light_attenuation = shadow_prd.attenuation;
    }

    // If not completely shadowed, light the hit point
    if( fmaxf(light_attenuation) > 0.0f ) {
      float3 Lc = light.color * light_attenuation;

      result += p_Kd * nDl * Lc;

      float3 H = normalize(L - ray.direction);
      float nDh = dot( p_normal, H );
      if(nDh > 0) {
        float power = pow(nDh, p_phong_exp);
        result += p_Ks * power * Lc;
      }
    }
  }

  if( fmaxf( p_reflectivity ) > 0 ) {

    // ray tree attenuation
    PerRayData_radiance_differentials new_prd;             
    new_prd.importance = prd_radiance.importance * optix::luminance( p_reflectivity );
    new_prd.depth = prd_radiance.depth + 1;

    // reflection ray
    if( new_prd.importance >= 0.01f && new_prd.depth <= max_depth) {
      float3 R = reflect( ray.direction, p_normal );
      optix::Ray refl_ray( hit_point, R, radiance_ray_type, scene_epsilon );

      // transfer differentials
      new_prd.origin_dx = differential_transfer_origin(prd_radiance.origin_dx, prd_radiance.direction_dx, t_hit, ray.direction, p_normal);
      new_prd.origin_dy = differential_transfer_origin(prd_radiance.origin_dy, prd_radiance.direction_dy, t_hit, ray.direction, p_normal);
      new_prd.direction_dx = prd_radiance.direction_dx;
      new_prd.direction_dy = prd_radiance.direction_dy;

      // reflect differentials
      new_prd.origin_dx = new_prd.origin_dx;
      new_prd.origin_dy = new_prd.origin_dy;
      new_prd.direction_dx = differential_reflect_direction(new_prd.origin_dx, new_prd.direction_dx, dNdP, ray.direction, p_normal);
      new_prd.direction_dy = differential_reflect_direction(new_prd.origin_dy, new_prd.direction_dy, dNdP, ray.direction, p_normal);


      rtTrace(top_object, refl_ray, new_prd);
      result += p_reflectivity * new_prd.result;
    }
  }
  
  // pass the color back up the tree
  prd_radiance.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
  phongShadowed();
}


RT_PROGRAM void closest_hit_radiance()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

  float3 ffnormal     = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  phongShade( Kd, Ka, Ks, ffnormal, phong_exp, reflectivity );
}
