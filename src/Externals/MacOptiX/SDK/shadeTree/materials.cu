
/*
 * Copyright (c) 2012 NVIDIA Corporation.  All rights reserved.
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

#include <optix_world.h>
#include "commonStructs.h"
#include "helpers.h"
#include "shader_defs.h"

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

using namespace optix;

rtBuffer<BasicLight>      lights;
rtDeclareVariable(float3, ambient_light_color, , ) = {0.f, 0.f, 0.f};
rtDeclareVariable(float3, specular_color, , ) = {1.f, 1.f, 1.f};
rtDeclareVariable(float,  phong_exp, , ) = 40.f;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 
rtDeclareVariable(float3, shading_tangent,  attribute shading_tangent, ); 
rtDeclareVariable(float2, uv, attribute uv, ); 


rtCallableProgram(float3, colorShader, (ShadingState const) );
rtCallableProgram(float3, normalShader, (ShadingState const, float3, float3 ) );



RT_PROGRAM void shade_tree_material()
{
  const float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  const float3 world_shading_tangent  = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_tangent ) );

  //float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  const float revn = copysignf( 1.f, dot(-ray.direction, world_geometric_normal));
  float3 ffnormal = world_shading_normal * revn;


  float3 hit_point = ray.origin + t_hit * ray.direction;

  ShadingState state;
  //state.uv = uv;
  state = uv;

  // bump 
  ffnormal = normalShader( state, ffnormal, world_shading_tangent * revn );

  float3 result = make_float3(0.f);
  const unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    BasicLight light = lights[i];
    const float3 L = optix::normalize(light.pos - hit_point);
    const float nDl = optix::dot( ffnormal, L);
    if (nDl > 0.f) {     
      const float3 color_from_shader = colorShader( state );
      result += nDl * color_from_shader * light.color;

      float3 H = optix::normalize(L - ray.direction);
      float nDh = optix::dot( ffnormal, H );
      if(nDh > 0.f) {
        float power = pow(nDh, phong_exp);
        float specular_factor = 1.f - 0.4f * luminance(color_from_shader); // play with specular intensity
        result += specular_color * power * light.color * specular_factor;
      }
    }
  }  
  const float3 color_from_shader = colorShader( state );
  prd.result = result + color_from_shader * ambient_light_color;
}



RT_PROGRAM void flat_modulated()
{
  ShadingState state;
  //state.uv = uv;
  state = uv;
  const float3 color_from_shader = colorShader( state );
  prd.result = color_from_shader;
}


