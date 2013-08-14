
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
#include <optix_math.h>
#include "helpers.h"
#include "mcmc_sampler.h"
#include "sunsky.h" 
#include "random.h"
#include "commonStructs.h"

rtDeclareVariable(float,      sun_scale, , );
rtDeclareVariable(float,      sky_scale, , );
rtDeclareVariable(float,      solid_angle, , ) = 6.0e-5f * 100.0f;

using namespace optix;

struct PerRayData_mcmcsampler
{
  float3 result;
  float3 radiance;
  float3 attenuation;
  float3 origin;
  float3 direction;
  int depth;
  int countEmitted;
  int done;
  MarkovChainState mcs;
  uint2 seed;
};

struct PerRayData_mcmcsampler_shadow
{
  bool inShadow;
};

// Scene wide
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(unsigned int, initial_sampling, , );
rtDeclareVariable(float3,       directional_light, , );
rtDeclareVariable(float3,       directional_light_col, , );
rtDeclareVariable(uint,         sampler_type, , );

// For camera
rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(unsigned int,  frame_number, , );
rtDeclareVariable(unsigned int,  frame_width, , );
rtDeclareVariable(unsigned int,  frame_height, , );
rtDeclareVariable(unsigned int,  num_chains, , );
rtBuffer<float4, 2>              output_buffer; 

rtBuffer<uint2, 1>               seed_buffer;
rtBuffer<uint2, 1>               init_seed_buffer;
rtBuffer<float4, 2>              accumulation_buffer;
rtBuffer<MarkovChainState, 1>    state_buffer; 

rtDeclareVariable(unsigned int,  mcmctrace_ray_type, , );
rtDeclareVariable(unsigned int,  mcmctrace_shadow_ray_type, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, ); 

rtDeclareVariable(PerRayData_mcmcsampler, current_prd, rtPayload, );

rtDeclareVariable(optix::Ray, ray,          rtCurrentRay, );
rtDeclareVariable(float,      t_hit,        rtIntersectionDistance, );
rtDeclareVariable(uint,       launch_index, rtLaunchIndex, );

rtDeclareVariable(float3,     bg_color, , );

// will be deleted once the new one is checked in
template<unsigned int N>
static __host__ __device__ __inline__ void tea_n( unsigned int& val0, unsigned int& val1 )
{
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    val0 += ((val1<<4)+0xa341316c)^(val1+s0)^((val1>>5)+0xc8013ea4);
    val1 += ((val0<<4)+0xad90777d)^(val0+s0)^((val0>>5)+0x7e95761e);
  }
}

static __device__ __inline__ float rnd(uint2 &n)
{
  tea_n<8>(n.x, n.y);
  return float(n.x + 1.0f) / float(0xffffffff + 1.0f);
}

// the importance function for the MCMC sampler
static __device__ __inline__ float importance(const float3 rgb)
{
  return fmaxf(rgb);
}

// mutate the value by the offset
static __device__ __inline__ float mutate(const float value, const float offset)
{
  float temp = value + offset;
  return temp - floor(temp);
}

// folding a function (put mirrored copies of a function to avoid false discontinuities)
static __device__ __inline__ float fold(const float value)
{
  return fminf(value, 1.0f - value) * 2.0f;
}

// generate a random number from the normal distribution
static __device__ __inline__ float gaussian(uint2 &seed)
{
  return sqrtf(-2.0f * logf(rnd(seed))) * cos(2.0f * M_PIf * rnd(seed));
}
//-----------------------------------------------------------------------------
//
//  Camera program -- main ray tracing loop
//
//-----------------------------------------------------------------------------

RT_PROGRAM void trace_camera()
{
  uint2 seed = seed_buffer[launch_index];
  
  if (sampler_type != SamplerUniform)
  {
    if (frame_number == 1) seed = init_seed_buffer[launch_index];
  }

  PerRayData_mcmcsampler prd;
  MarkovChainState original_state;

  if (frame_number > 1)
  {
    //-----------------------------------------------------------------------------
    // mutation
    //-----------------------------------------------------------------------------

    // backup original state
    original_state = state_buffer[launch_index];

    if (sampler_type == SamplerRandomWalk)
    {
      // random walk Markov chain Monte Carlo
      
      // generate uniform random direction
      float dir[RNDS_COUNT];
      float dir_len = 0.0f;
      for (int i = 0; i < RNDS_COUNT; i++)
      {
        dir[i] = gaussian(seed);
        dir_len += dir[i] * dir[i];
      }
      
      // scale it by the mutation size
      const float gamma = 3e-2f;
      float mutation_size = gamma * tanf(M_PIf * (rnd(seed) - 0.5f));
      dir_len = mutation_size / sqrtf(dir_len);
      
      // mutation
      for (int i = 0; i < RNDS_COUNT; i++)
      {
        prd.mcs.rnds[i] = mutate(original_state.rnds[i], dir_len * dir[i]);
      }
    }
    else if (sampler_type == SamplerUniform)
    {
      // uniform random sampling
      for (int i = 0; i < RNDS_COUNT; i++)
      {
        prd.mcs.rnds[i] = rnd(seed);
      }
    }
  }
  else
  { 
    //-----------------------------------------------------------------------------
    // initialization
    //-----------------------------------------------------------------------------

    for (unsigned int i = 0; i < RNDS_COUNT; i++)
    {
      prd.mcs.rnds[i] = rnd(seed);
    }

    prd.mcs.f = 0.0f;
    prd.mcs.c = make_float3(0.0f);

    original_state = prd.mcs;
  }

  //-----------------------------------------------------------------------------
  // compute the path
  //-----------------------------------------------------------------------------
  size_t2 screen = output_buffer.size();
  float2 screenpos = make_float2( fold(prd.mcs.rnds[0]), fold(prd.mcs.rnds[1]) );
  float2 pixel = 2.0f * screenpos - 1.0f;
  uint2 pixel_index;

  float3 result = make_float3( 0.0f );

  // compute the eye ray
  float3 ray_origin = eye;
  float3 ray_direction = normalize( pixel.x * U + pixel.y * V + W );

  // setup the payload
  prd.result = make_float3( 0.f );
  prd.attenuation = make_float3( 1.f );
  prd.countEmitted = true;
  prd.done = false;
  prd.depth = 0;
  prd.seed = seed;

  // trace further bounces
  for (int i = 1; i <= MAX_BOUNCE; i++)
  {
    Ray ray = make_Ray( ray_origin, ray_direction, mcmctrace_ray_type, scene_epsilon, RT_DEFAULT_MAX );
    rtTrace( top_object, ray, prd );
    prd.result += prd.radiance * prd.attenuation;
    if (prd.done) break;

    prd.depth++;
    ray_origin = prd.origin;
    ray_direction = prd.direction;
  }
  seed = prd.seed; 
  result = prd.result;

  float c = importance(prd.result);
  if (sampler_type == SamplerUniform) c = 1.0f;

  if (frame_number > 1)
  {
    //-----------------------------------------------------------------------------
    // accept or reject the mutated sample
    //-----------------------------------------------------------------------------
    float acceptance = min( 1.0f, c / original_state.f );
    if (original_state.f == 0.0f) acceptance = 1.0f;

    // splat the mutated sample
    if (c != 0.0f)
    {
      result = prd.result / c;
    }
    else
    {
      result = make_float3( 0.0f );
    }
    screenpos = make_float2( fold(prd.mcs.rnds[0]), fold(prd.mcs.rnds[1]) );
    pixel_index = make_uint2( frame_width * screenpos.x, frame_height * screenpos.y );
    pixel_index.x %= frame_width;
    pixel_index.y %= frame_height;
    float3 pixel_color = result * acceptance; 
    atomicAdd( (float*)&accumulation_buffer[pixel_index].x, pixel_color.x );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].y, pixel_color.y );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].z, pixel_color.z );

    // splat the original sample
    result = original_state.c;
    screenpos = make_float2( fold(original_state.rnds[0]), fold(original_state.rnds[1]) );
    pixel_index = make_uint2( frame_width * screenpos.x, frame_height * screenpos.y );
    pixel_index.x %= frame_width;
    pixel_index.y %= frame_height;
    pixel_color = result * (1.0f - acceptance); 
    atomicAdd( (float*)&accumulation_buffer[pixel_index].x, pixel_color.x );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].y, pixel_color.y );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].z, pixel_color.z );

    if (acceptance > rnd(seed))
    {
      // accept mutation
      prd.mcs.f = c;
      if (prd.mcs.f != 0.0f)
      {
        prd.mcs.c = prd.result / prd.mcs.f;
      }
      else
      {
        prd.mcs.c = make_float3(0.0f);
      }
    }
    else
    {
      // reject mutation
      prd.mcs = original_state;
    }
  }
  else
  {
    //-----------------------------------------------------------------------------
    // initialization
    //-----------------------------------------------------------------------------
    prd.mcs.f = c;
    if (prd.mcs.f != 0.0f)
    {
      prd.mcs.c = prd.result / prd.mcs.f;
    }
    else
    {
      prd.mcs.c = make_float3(0.0f);
    }

    result = prd.mcs.c * (initial_sampling == 1 ? prd.mcs.f : 1.0f);
    screenpos = make_float2( fold(prd.mcs.rnds[0]), fold(prd.mcs.rnds[1]) );
    pixel_index = make_uint2( frame_width * screenpos.x, frame_height * screenpos.y );
    pixel_index.x %= frame_width;
    pixel_index.y %= frame_height;
    float3 pixel_color = result;
    atomicAdd( (float*)&accumulation_buffer[pixel_index].x, pixel_color.x );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].y, pixel_color.y );
    atomicAdd( (float*)&accumulation_buffer[pixel_index].z, pixel_color.z );
  }

  if (sampler_type != SamplerUniform)
  {
    if (frame_number != 1) seed_buffer[launch_index] = seed;
  }
  else
  {
    seed_buffer[launch_index] = seed;
  }

  state_buffer[launch_index] = prd.mcs;
}


rtDeclareVariable(float,  phong_exp, , );
rtDeclareVariable(float3, emissive, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtTextureSampler<float4, 2> ambient_map;
rtTextureSampler<float4, 2> diffuse_map;
rtTextureSampler<float4, 2> specular_map;
rtBuffer<TriangleLight, 1>  light_buffer;

enum EMaterialType
{
  MT_Matte,
  MT_Metal,
  MT_Glass,
  MT_GlossyMetal,
  MT_Light
};

RT_PROGRAM void uber_material()
{
  // ****************************************************************************************************** 
  // in general, using uber material is not necessary in OptiX, but was used for convenience in this sample
  // ******************************************************************************************************

  // collect hitpoint info
  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );
  float2 uv                     = make_float2( texcoord );

  float3 hitpoint = ray.origin + t_hit * ray.direction;
  current_prd.origin = hitpoint;
  current_prd.radiance = make_float3( 0.0f );

  // obtain OBJ file material parameters
  float3 Kd = make_float3( tex2D(diffuse_map,  uv.x, uv.y) );
  float3 Ks = make_float3( tex2D(specular_map, uv.x, uv.y) );
  float3 Ka = make_float3( tex2D(ambient_map,  uv.x, uv.y) );

  // set the material type
  EMaterialType MaterialType;
  if (fmaxf(emissive) > 0.0f)
  {
    MaterialType = MT_Light;
  }
  else
  {
    if (fmaxf(Ks) > 0.0f)
    {
      if (phong_exp == 0.0f)
      {
        MaterialType = MT_Glass;
      }
      else
      {
        MaterialType = MT_Metal;
      }
    }
    else
    {
      if (phong_exp == 0.0f)
      {
        MaterialType = MT_Matte;
      }
      else
      {
        MaterialType = MT_GlossyMetal;
      }
    }
  }

  // execute the uber shader
  if (MaterialType == MT_Light)
  {
    // --------------------------------
    // light
    // --------------------------------
    current_prd.radiance = current_prd.countEmitted? emissive * 2.0f: make_float3(0.f);
    
    // the back face is black
    if (dot(world_geometric_normal, ray.direction) > 0.0f) current_prd.radiance = make_float3(0.0f); 

    // end the path trace
    current_prd.done = true;
  }
  else if (MaterialType == MT_Metal)
  {
    // --------------------------------
    // metal
    // --------------------------------
    current_prd.origin = hitpoint;
    current_prd.countEmitted = true;
    current_prd.radiance = make_float3(0.0f);

    // specular reflection
    current_prd.direction = reflect(ray.direction, world_shading_normal);
    current_prd.attenuation = current_prd.attenuation * Kd;
  }
  else if (MaterialType == MT_Glass)
  {
    // --------------------------------
    // glass
    // --------------------------------
    current_prd.origin = hitpoint;
    current_prd.countEmitted = true;
    current_prd.radiance = make_float3(0.0f);  

    const float n1 = 1.0f;
    const float n2 = 1.5f;

    float3 reflect_dir = reflect(ray.direction, world_shading_normal);
    bool into = (dot(ray.direction, world_shading_normal) < 0.0f);
    float n_ratio = into ? (n1 / n2) : (n2 / n1);
    float dDn = dot(ray.direction, ffnormal);
    float cos2t = 1.0f - n_ratio * n_ratio * (1.0f - dDn * dDn);

    if (cos2t < 0.0f)
    {
      // total internal reflection
      current_prd.direction = reflect_dir;
    }
    else
    {
      float3 refract_dir = normalize( ray.direction * n_ratio - ffnormal * (dDn * n_ratio + sqrtf(cos2t))) ;
      float cost = dot(refract_dir, world_shading_normal);

      // Fresnel reflectance
      float Rs = (n_ratio * abs(dDn) - abs(cost)) / (n_ratio * abs(dDn) + abs(cost));
      float Rp = (n_ratio * abs(cost) - abs(dDn)) / (n_ratio * abs(cost) + abs(dDn));
      float Re = (Rs * Rs + Rp * Rp) * 0.5f;

      // Russian roulette
      float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 0];
      if (fold(z1) < Re)
      {
        // reflection
        current_prd.direction = reflect_dir;
      }
      else
      {
        // refraction
        current_prd.attenuation = current_prd.attenuation * Kd;
        current_prd.direction = refract_dir;
      }
    }
  }
  else if (MaterialType == MT_Matte)
  {
    // --------------------------------
    // matte
    // --------------------------------

    // calculate diffuse reflected vector
    float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 0];
    float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 1];
    float z3 = powf(fold(z1), 2.0f / (1.0f + 1.0f));
    float sintheta = sqrtf(1.0f - z3);
    float costheta = sqrtf(z3);
    float phi = (2.0f * M_PIf) * z2;
    float cosphi = cosf(phi) * sintheta;
    float sinphi = sinf(phi) * sintheta;
    float3 v1, v2;
    create_onb(ffnormal, v1, v2);
    current_prd.direction = ffnormal * costheta + v1 * cosphi + v2 * sinphi;

    current_prd.attenuation = current_prd.attenuation * Kd;
    current_prd.countEmitted = false; 

    // --------------------------------
    // compute direct light
    // --------------------------------
    unsigned int num_lights = light_buffer.size();
    float3 result = make_float3(0.0f);

    if (num_lights == 0)
    {
      // --------------------------------
      // directional light source
      // --------------------------------

      // light direction
      float3 L = normalize(directional_light);

      // perturb the direction based on given solid angle
      {
        float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 2];
        float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 3];

        float cosmax = 1.0f - solid_angle / (2.0f * M_PIf);
        float z3 = 1.0f - fold(z2) * (1.0f - cosmax);

        float sintheta = sqrtf(1.0f - z3 * z3);
        float costheta = z3 * z3;
        float phi = (2.0f * M_PIf) * z1;
        float cosphi = cosf(phi) * sintheta;
        float sinphi = sinf(phi) * sintheta;
        float3 v1, v2;
        create_onb(L, v1, v2);
        L = L * costheta + v1 * cosphi + v2 * sinphi;
      }

      float nDl = dot( ffnormal, L );

      // cast shadow ray
      if ( nDl > 0.0f )
      {
        PerRayData_mcmcsampler_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, mcmctrace_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if (!shadow_prd.inShadow)
        {
          float weight = nDl;
          result += directional_light_col * weight;
        }
      }
    }
    else
    {
      // --------------------------------
      // triangle light source
      // --------------------------------

      // it assumes areas of light sources are the same...
      int i = int(num_lights * current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 4]);

      // uniform sampling of a point on the light source
      TriangleLight light = light_buffer[i];
      float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 2];
      float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 3];
      float alpha = 1.0f - sqrt(fold(z1));
      float beta = (1.0f - fold(z2)) * sqrt(fold(z1));
      float gamma = fold(z2) * sqrt(fold(z1));
      float3 light_pos = light.v3 * gamma + light.v1 * alpha + light.v2 * beta;

      float Ldist = length(light_pos - hitpoint);
      float3 L = normalize(light_pos - hitpoint);
      float nDl = dot( ffnormal, L );
      float LnDl = dot( light.normal, L );
      float A = length(cross(light.v2 - light.v3, light.v1 - light.v3));

      // cast shadow ray
      if ( nDl > 0.0f && LnDl > 0.0f ) 
      {
        PerRayData_mcmcsampler_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, mcmctrace_shadow_ray_type, scene_epsilon, Ldist );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if (!shadow_prd.inShadow)
        {
          float weight = nDl * LnDl / (M_PIf*Ldist*Ldist) * A;
          result += num_lights * light.emission * weight;
        }
      }
    }

    current_prd.radiance = result;
  }
  else if (MaterialType == MT_GlossyMetal)
  {
    // --------------------------------
    // glossy metal
    // --------------------------------

    // calculate the half vector
    float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 0];
    float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 1];
    float z3 = powf(fold(z1), 2.0f / (phong_exp + 1.0f));
    float sintheta = sqrtf(1.0f - z3);
    float costheta = sqrtf(z3);
    float phi = (2.0f * M_PIf) * z2;
    float cosphi = cosf(phi) * sintheta;
    float sinphi = sinf(phi) * sintheta;
    float3 v1, v2;
    create_onb(ffnormal, v1, v2);

    float3 h = ffnormal * costheta + v1 * cosphi + v2 * sinphi;

    current_prd.direction = reflect(ray.direction, h);
    if (dot(current_prd.direction, ffnormal) < 0.0) current_prd.direction = current_prd.direction;

    current_prd.attenuation = current_prd.attenuation * Kd;
    current_prd.countEmitted = false; 

    // --------------------------------
    // compute direct light
    // --------------------------------
    unsigned int num_lights = light_buffer.size();
    float3 result = make_float3(0.0f);

    if (num_lights == 0)
    {
      // --------------------------------
      // directional light source
      // --------------------------------

      // light direction
      float3 L = normalize(directional_light);

      // perturb the direction based on given solid angle
      {
        float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 2];
        float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 3];

        float cosmax = 1.0f - solid_angle / (2.0f * M_PIf);
        float z3 = 1.0f - fold(z2) * (1.0f - cosmax);

        float sintheta = sqrtf(1.0f - z3 * z3);
        float costheta = z3 * z3;
        float phi = (2.0f * M_PIf) * z1;
        float cosphi = cosf(phi) * sintheta;
        float sinphi = sinf(phi) * sintheta;
        float3 v1, v2;
        create_onb(L, v1, v2);
        L = L * costheta + v1 * cosphi + v2 * sinphi;
      }

      float nDl = dot( ffnormal, L );

      // cast shadow ray
      if ( nDl > 0.0f )
      {
        PerRayData_mcmcsampler_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, mcmctrace_shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if (!shadow_prd.inShadow)
        {
          float3 h = normalize(-ray.direction + shadow_ray.direction);
          float nDotH = dot(ffnormal, h);
          float weight = powf(max(nDotH, 0.0f), phong_exp) * (phong_exp + 2.0f) / (M_PIf * 2.0f);
          result += directional_light_col * weight;
        }
      }
    }
    else
    {
      // --------------------------------
      // triangle light source
      // --------------------------------

      // it assumes areas of light sources are the same...
      int i = int(num_lights * current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 4]);

      // uniform sampling of a point on the light source
      TriangleLight light = light_buffer[i];
      float z1 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 2];
      float z2 = current_prd.mcs.rnds[current_prd.depth * 5 + 2 + 3];
      float alpha = 1.0f - sqrt(fold(z1));
      float beta = (1.0f - fold(z2)) * sqrt(fold(z1));
      float gamma = fold(z2) * sqrt(fold(z1));
      float3 light_pos = light.v3 * gamma + light.v1 * alpha + light.v2 * beta;

      float Ldist = length(light_pos - hitpoint);
      float3 L = normalize(light_pos - hitpoint);
      float nDl = dot( ffnormal, L );
      float LnDl = dot( light.normal, L );
      float A = length(cross(light.v2 - light.v3, light.v1 - light.v3));

      // cast shadow ray
      if ( nDl > 0.0f && LnDl > 0.0f ) 
      {
        PerRayData_mcmcsampler_shadow shadow_prd;
        shadow_prd.inShadow = false;
        Ray shadow_ray = make_Ray( hitpoint, L, mcmctrace_shadow_ray_type, scene_epsilon, Ldist );
        rtTrace(top_object, shadow_ray, shadow_prd);

        if (!shadow_prd.inShadow)
        {
           float3 h = normalize(-ray.direction + shadow_ray.direction);
          float nDotH = dot(ffnormal, h);
          float weight = LnDl / (M_PIf*Ldist*Ldist) * A * powf(max(nDotH , 0.0f), phong_exp) * (phong_exp + 2.0f) / (M_PIf * 2.0f);

          result += num_lights * light.emission * weight;
        }
      }
    }
    
    current_prd.radiance = result;
  }
}

//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void exception()
{
  state_buffer[launch_index] = current_prd.mcs;
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

  // use the sky light model
  if (light_buffer.size() == 0)
  {
    // only above the horizon
    if (dot( ray.direction, sky_up ) > 0.0f)
    {
      current_prd.radiance = querySkyModel(  false, ray.direction ) * sky_scale;
    }

    // if the ray direction is within the sun's solid angle, return radiance of the sun
    float3 L = normalize(directional_light);
    float nDl = dot( ray.direction, L );
    if ( nDl > (1.0f - solid_angle / (2.0f * M_PIf)) )
    {
      // avoid double counting illumination
      current_prd.radiance = current_prd.countEmitted ? directional_light_col * (M_PIf / solid_angle) : make_float3(0.0f);
    }
  }
}

rtDeclareVariable(PerRayData_mcmcsampler_shadow, current_prd_shadow, rtPayload, );

RT_PROGRAM void shadow()
{
  float2 uv = make_float2(texcoord);
  float3 Ka = make_float3(tex2D(ambient_map, uv.x, uv.y));

  // no direct shadow from light sources
  if (fmaxf(emissive) == 0.0f)
  {
    current_prd_shadow.inShadow = true;
    rtTerminateRay();
  }
}
