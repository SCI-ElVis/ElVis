
/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
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
#include "path_tracer.h"
#include "random.h"

using namespace optix;

struct PerRayData_pathtrace
{
  float3 result;
  float3 radiance;
  float3 attenuation;
  float3 origin;
  float3 direction;
  unsigned int seed;
  int depth;
  int countEmitted;
  int done;
  int inside;
};

struct PerRayData_pathtrace_shadow
{
  bool inShadow;
};

struct PerRayData_pathtrace_bsdf_shadow
{
  float3 radiance;
  float t;
  int lgt_idx;
};

// Scene wide
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(rtObject,      top_object, , );

// For camera
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  sqrt_num_samples, , );
rtBuffer<float4, 2>              output_buffer;
rtDeclareVariable(float,        Y_log_av, , );
rtDeclareVariable(float,        Y_max, , );
rtDeclareVariable(int,  sampling_strategy, , );


rtBuffer<float3, 2>              hdr_buffer;
rtBuffer<ParallelogramLight>     lights;

rtDeclareVariable(unsigned int,  pathtrace_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_shadow_ray_type, , );
rtDeclareVariable(unsigned int,  pathtrace_bsdf_shadow_ray_type, , );
rtDeclareVariable(unsigned int,  rr_begin_depth, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 
rtDeclareVariable(float3, texcoord,         attribute texcoord, ); 
rtDeclareVariable(int, lgt_idx, attribute lgt_idx, ); 

rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );


// For miss program
rtDeclareVariable(float3,       bg_color, , );

//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

RT_PROGRAM void pathtrace_camera()
{
  size_t2 screen = output_buffer.size();

  float2 inv_screen = 1.0f/make_float2(screen) * 2.f;
  float2 pixel = (make_float2(launch_index)) * inv_screen - 1.f;

  float2 jitter_scale = inv_screen / sqrt_num_samples;
  unsigned int samples_per_pixel = sqrt_num_samples*sqrt_num_samples;
  float3 result = make_float3(0.0f);

  unsigned int seed = tea<16>(screen.x*launch_index.y+launch_index.x, frame_number);
  do {
    unsigned int x = samples_per_pixel%sqrt_num_samples;
    unsigned int y = samples_per_pixel/sqrt_num_samples;
    float2 jitter = make_float2(x-rnd(seed), y-rnd(seed));
    float2 d = pixel + jitter*jitter_scale;
    float3 ray_origin = eye;
    float3 ray_direction = normalize(d.x*U + d.y*V + W);

    PerRayData_pathtrace prd;
    prd.result = make_float3(0.f);
    prd.attenuation = make_float3(1.f);
    prd.countEmitted = true;
    prd.done = false;
    prd.inside = false;
    prd.seed = seed;
    prd.depth = 0;

    for(;;) {
      Ray ray = make_Ray(ray_origin, ray_direction, pathtrace_ray_type, scene_epsilon, RT_DEFAULT_MAX);
      rtTrace(top_object, ray, prd);
      if(prd.done) {
        prd.result += prd.radiance * prd.attenuation;
        break;
      }

      // RR
      if(prd.depth >= rr_begin_depth){
        float pcont = fmaxf(prd.attenuation);
        if(rnd(prd.seed) >= pcont)
          break;
        prd.attenuation /= pcont;
      }
      prd.depth++;
      prd.result += prd.radiance * prd.attenuation;
      ray_origin = prd.origin;
      ray_direction = prd.direction;
    } // eye ray

    result += prd.result;
    seed = prd.seed;
  } while (--samples_per_pixel);

  float3 pixel_color = result/(sqrt_num_samples*sqrt_num_samples);

  if (frame_number > 1)
  {
    float a = 1.0f / (float)frame_number;
    float b = ((float)frame_number - 1.0f) * a;
    hdr_buffer[launch_index] = a * pixel_color + b * hdr_buffer[launch_index];
  }
  else
    hdr_buffer[launch_index] = pixel_color;

  output_buffer[launch_index] = make_float4(tonemap(hdr_buffer[launch_index], Y_log_av, Y_max), 0.0f);
}

rtDeclareVariable(float3,        emission_color, , );

rtTextureSampler<float4, 2> emit_tex;

RT_PROGRAM void diffuseEmitter()
{
  current_prd.radiance = current_prd.countEmitted? emission_color : make_float3(0.f);
  current_prd.done = true;
}

RT_PROGRAM void diffuseTexEmitter()
{
  current_prd.radiance = current_prd.countEmitted ? 
    (make_float3( tex2D(emit_tex, texcoord.x, texcoord.y) ) * emission_color) : make_float3(0.f);
  current_prd.done = true;
}

rtDeclareVariable(PerRayData_pathtrace_bsdf_shadow, bsdf_shadow_prd, rtPayload, );

RT_PROGRAM void diffuseMISEmitter()
{
  bsdf_shadow_prd.radiance = emission_color;
  bsdf_shadow_prd.lgt_idx = lgt_idx; 
  bsdf_shadow_prd.t = t_hit;
}

RT_PROGRAM void diffuseTexMISEmitter()
{
  bsdf_shadow_prd.radiance = emission_color;
  bsdf_shadow_prd.radiance *= make_float3( tex2D(emit_tex, texcoord.x, texcoord.y) );
  bsdf_shadow_prd.lgt_idx = lgt_idx;
  bsdf_shadow_prd.t = t_hit;
}

rtDeclareVariable(float3,        diffuse_color, , );

enum
{
  SamplingStrategyBSDF,
  SamplingStrategyLIGHT,
  SamplingStrategyMIS
};

RT_PROGRAM void diffuse()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

  float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hitpoint = ray.origin + t_hit * ray.direction;
  current_prd.origin = hitpoint;

  
  float z1=rnd(current_prd.seed);
  float z2=rnd(current_prd.seed);
  float3 p;
  cosine_sample_hemisphere(z1, z2, p);
  float3 v1, v2;
  createONB(ffnormal, v1, v2);
  current_prd.direction = v1 * p.x + v2 * p.y + ffnormal * p.z;

  // Compute direct light...
  // Or shoot one...
  float3 result = make_float3(0.0f);

  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    ParallelogramLight light = lights[i];
      
    float A = length(cross(light.v1, light.v2));

    if (sampling_strategy == SamplingStrategyMIS) {
      float z1=rnd(current_prd.seed);
      float z2=rnd(current_prd.seed);
      float sintheta2=z1;
      float sintheta=sqrtf(sintheta2);
      float costheta=sqrtf(1.f-sintheta2);
      float phi=(2.f*M_PIf)*z2;
      float cosphi=cosf(phi)*sintheta;
      float sinphi=sinf(phi)*sintheta;

      float3 direction = ffnormal * costheta + v1 * cosphi + v2 * sinphi;
      float bsdf_pdf = costheta / M_PIf;
      float3 bsdf_val = diffuse_color / M_PIf;


      if (bsdf_pdf > 0.0f) {
        PerRayData_pathtrace_bsdf_shadow new_prd;
        new_prd.radiance = make_float3(0.0f);
        new_prd.t = -1.0f;

        Ray new_ray = make_Ray(hitpoint, direction, pathtrace_bsdf_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, new_ray, new_prd);

        // PDF of the light in solid angle measure 
        float lgt_pdf = 0.0f;
        float LnDl = max(-dot( light.normal, direction ), 0.0f);

        if (LnDl > 0.0f) {
          lgt_pdf = new_prd.lgt_idx == i ? (new_prd.t * new_prd.t / (LnDl * A)) : 0.0f;
        }
        else
          new_prd.radiance = make_float3(0.0f);

        // power heuristic weight
        float w = bsdf_pdf * bsdf_pdf / (bsdf_pdf * bsdf_pdf + lgt_pdf * lgt_pdf);

        if (sampling_strategy == SamplingStrategyMIS)
          result += new_prd.radiance * bsdf_val * w * costheta / bsdf_pdf;
        else
          result += new_prd.radiance * bsdf_val * costheta / bsdf_pdf;
      }
    }
    
    if (sampling_strategy == SamplingStrategyLIGHT || sampling_strategy == SamplingStrategyMIS) {
      float z1 = rnd(current_prd.seed);
      float z2 = rnd(current_prd.seed);
      float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

      float Ldist = length(light_pos - hitpoint);
      float3 L = normalize(light_pos - hitpoint);
      float nDl = dot( ffnormal, L );
      float LnDl = max(-dot( light.normal, L ), 0.0f);

      // cast shadow ray
      if ( nDl > 0.0f && LnDl > 0.0f ) {
        PerRayData_pathtrace_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if(!shadow_prd.inShadow) {
          float bsdf_pdf = nDl / M_PIf;
          float lgt_pdf = Ldist * Ldist / (LnDl * A);

          float3 emission = light.emission;
          if (light.textured)
            emission *= make_float3( tex2D(emit_tex, z1, z2) );

          // power heuristic weight
          float w = lgt_pdf * lgt_pdf / (bsdf_pdf * bsdf_pdf + lgt_pdf * lgt_pdf);
          float3 bsdf_val = diffuse_color / M_PIf;
          if (lgt_pdf > 0.0f) {
            if (sampling_strategy == SamplingStrategyMIS)
              result += emission * bsdf_val * w * nDl / lgt_pdf;
            else
              result += emission * bsdf_val * nDl / lgt_pdf;
          }
        }
      }
    }
  }  // loop over lights
    
  current_prd.result += current_prd.attenuation * result;
  current_prd.radiance = make_float3(0.0f);
  current_prd.attenuation = current_prd.attenuation * diffuse_color; // use the diffuse_color as the diffuse response
  current_prd.countEmitted = sampling_strategy == SamplingStrategyBSDF ? true : false;
}

rtDeclareVariable(float,  exponent,      , );
rtDeclareVariable(float3, glossy_color,  , );

RT_PROGRAM void glossy()
{
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );

  float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hitpoint = ray.origin + t_hit * ray.direction;
  current_prd.origin = hitpoint;
 
  float2 sample;
  sample.x = rnd(current_prd.seed);
  sample.y = rnd(current_prd.seed);
  float bsdf_pdf;
  float bsdf_val;
  float3 dir_out = -ray.direction;
  float3 dir_r = reflect(ray.direction, ffnormal);

  float3 v1, v2;
  createONB(dir_r, v1, v2);
  current_prd.direction = sample_phong_lobe( sample, exponent, v1, v2, dir_r, bsdf_pdf, bsdf_val );

  // Compute direct light...
  // Or shoot one...
  float3 result = make_float3(0.0f);
  unsigned int num_lights = lights.size();
  for(int i = 0; i < num_lights; ++i) {
    ParallelogramLight light = lights[i];
    float A = length(cross(light.v1, light.v2));

    if (sampling_strategy == SamplingStrategyMIS) {
      float2 sample;
      sample.x = rnd(current_prd.seed);
      sample.y = rnd(current_prd.seed);
      float bsdf_pdf;
      float bsdf_val;
      float3 direction = sample_phong_lobe( sample, exponent, v1, v2, dir_r, bsdf_pdf, bsdf_val );
      float costheta = dot(direction, ffnormal);
      if (costheta <= 0.0f)
      {
        bsdf_pdf = 0.0f;
        bsdf_val = 0.0f;
      }

      if (bsdf_pdf > 0.0f) {
        PerRayData_pathtrace_bsdf_shadow new_prd;
        new_prd.radiance = make_float3(0.0f);
        new_prd.t = -1.0f;

        Ray new_ray = make_Ray(hitpoint, direction, pathtrace_bsdf_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, new_ray, new_prd);

        if (new_prd.t <= 0.0f) 
          new_prd.radiance = make_float3(0.0f);  // background light

        // PDF of the light in solid angle measure 
        float lgt_pdf = 0.0f;
        float LnDl = max(-dot( light.normal, direction ), 0.0f);

        if (LnDl > 0.0f) {
          lgt_pdf = new_prd.lgt_idx == i ? (new_prd.t * new_prd.t / (LnDl * A)) : 0.0f;
        }
        else
          new_prd.radiance = make_float3(0.0f);

        // power heuristic weight
        float w = bsdf_pdf * bsdf_pdf / (bsdf_pdf * bsdf_pdf + lgt_pdf * lgt_pdf);
        if (sampling_strategy == SamplingStrategyMIS)
          result += new_prd.radiance * glossy_color * (costheta * w * bsdf_val / bsdf_pdf);
        else
          result += new_prd.radiance * costheta * glossy_color * bsdf_val / bsdf_pdf;
      }
    }

    if (sampling_strategy == SamplingStrategyLIGHT || sampling_strategy == SamplingStrategyMIS) {
      float z1 = rnd(current_prd.seed);
      float z2 = rnd(current_prd.seed);
      float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

      float Ldist = length(light_pos - hitpoint);
      float3 L = normalize(light_pos - hitpoint);
      float nDl = dot( ffnormal, L );
      float LnDl = max(-dot( light.normal, L ), 0.0f);

      // cast shadow ray
      if ( nDl > 0.0f && LnDl > 0.0f ) {
        PerRayData_pathtrace_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, pathtrace_shadow_ray_type, scene_epsilon, Ldist );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if(!shadow_prd.inShadow) {
          float bdf_v;
          float bsdf_pdf = get_phong_lobe_pdf(exponent, ffnormal, dir_out, L, bdf_v);
          float3 bsdf_val = glossy_color * bdf_v;

          float lgt_pdf = Ldist * Ldist / (LnDl * A);
          float3 emission = light.emission;
          if (light.textured)
            emission *= make_float3( tex2D(emit_tex, z1, z2) );

          // power heuristic weight
          float w = lgt_pdf * lgt_pdf / (bsdf_pdf * bsdf_pdf + lgt_pdf * lgt_pdf);
          if (lgt_pdf > 0.0f) {
            if (sampling_strategy == SamplingStrategyMIS)
              result += emission * bsdf_val * w * nDl / lgt_pdf;
            else
              result += emission * bsdf_val * nDl / lgt_pdf;
          }
        }
      }
    }
  } // loop over lights

  current_prd.result += current_prd.attenuation * result;
  current_prd.radiance = make_float3(0.0f);
  if (bsdf_pdf > 0.f)
    current_prd.attenuation = current_prd.attenuation * glossy_color * bsdf_val / bsdf_pdf * 
                                                        fabs(dot(current_prd.direction, ffnormal));
  else {
    current_prd.done = true;
  }
  current_prd.countEmitted = sampling_strategy == SamplingStrategyBSDF ? true : false;
}

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
  output_buffer[launch_index] = make_float4(bad_color, 0.0f);
}

//-----------------------------------------------------------------------------
//
//  Miss program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void miss()
{
  current_prd.radiance = bg_color;
  current_prd.done = true;
}


rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
  current_prd_shadow.inShadow = true;
  rtTerminateRay();
}
