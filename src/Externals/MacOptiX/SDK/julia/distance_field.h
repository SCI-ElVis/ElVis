
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

#pragma once

#include <optixu/optixu_math_namespace.h>

//#define HD_DECL __host__ __device__
#define HD_DECL __device__

template<unsigned int max_iterations, typename Distance>
static __device__
  float sphere_trace(const optix::Ray &ray, Distance distance,
                     float3 &x,
                     const float epsilon = 1e-3f,
                     const float step_scale = 1.0f)
{
  float3 ray_direction = ray.direction;

  x = ray.origin + ray.tmin * ray_direction;

  float distance_from_current_point_to_ray_origin = ray.tmin;

  float dist;
  for(unsigned int i = 0; i < max_iterations; ++i)
  {
    dist = distance(x);

    if( dist < epsilon )
      break;

    dist *= step_scale;

    // step the current point along the ray and accumulate the distance from the origin
    x += dist * ray.direction;
    distance_from_current_point_to_ray_origin += dist;

    if( distance_from_current_point_to_ray_origin > ray.tmax )
      break;
  }

  return distance_from_current_point_to_ray_origin;
}


template<unsigned int max_iterations, typename Distance>
static __device__
  float adaptive_sphere_trace(const optix::Ray &ray, Distance distance,
                              float3 &x,
                              const float epsilon = 1e-3f,
                              const float max_epsilon = 1e-2f,
                              const float step_scale = 1.0f)
{
  float3 ray_direction = ray.direction;

  x = ray.origin + ray.tmin * ray_direction;

  float distance_from_current_point_to_ray_origin = ray.tmin;

  float dist;
  for(unsigned int i = 0; i < max_iterations; ++i)
  {
    dist = distance(x);

    if( dist < fminf(max_epsilon, epsilon * distance_from_current_point_to_ray_origin) )
      break;

    dist *= step_scale;

    // step the current point along the ray and accumulate the distance from the origin
    x += dist * ray.direction;
    distance_from_current_point_to_ray_origin += dist;

    if( distance_from_current_point_to_ray_origin > ray.tmax )
      break;
  }

  return distance_from_current_point_to_ray_origin;
}


template<typename SignedDistance>
static __device__
float3 grad(SignedDistance d, const float3 &x, const float eps = 1e-6f)
{
  float dx = d(x + make_float3(eps,    0,   0)) - d(x - make_float3(eps,   0,   0));
  float dy = d(x + make_float3(  0,  eps,   0)) - d(x - make_float3(  0, eps,   0));
  float dz = d(x + make_float3(  0,    0, eps)) - d(x - make_float3(  0,   0, eps));

  return make_float3(dx, dy, dz);
}


template<typename SignedDistance>
static __device__
float3 estimate_normal(const SignedDistance distance,
                       const float3 &x,
                       const float epsilon = 1e-3f)
{
  return normalize(grad(distance, x, epsilon));
}

template<unsigned int num_samples,
         typename Distance>
static HD_DECL
  float magic_ambient_occlusion(const float3 &x, const float3 &n,
                                const float del,
                                const float k,
                                Distance distance)
{
  float inv_power_of_two = 1.0f;
  float sum = 0;

  float4 step = make_float4(del * n, del);
  float4 current = make_float4(x, 0);

  for(unsigned int i = 0;
      i < num_samples;
      ++i, inv_power_of_two *= 0.5f, current += step)
  {
    sum += inv_power_of_two * (current.w - distance(make_float3(current)));
  }

  return optix::clamp(1.0f - k * sum, 0.0f, 1.0f);
}


template<typename Primitive>
  struct DistanceToPrimitive
{
  HD_DECL
  DistanceToPrimitive(void){}

  HD_DECL
  DistanceToPrimitive(Primitive prim):m_prim(prim){}

  HD_DECL
  float operator()(const float3 &x) const
  {
    return m_prim.distance(x);
  }

  Primitive m_prim;
};

template<typename Primitive>
static HD_DECL
  DistanceToPrimitive<Primitive> make_distance_to_primitive(Primitive prim)
{
  return DistanceToPrimitive<Primitive>(prim);
}


template<typename Primitive>
  struct SignedDistanceToPrimitive
{
  HD_DECL
  SignedDistanceToPrimitive(void){}

  HD_DECL
  SignedDistanceToPrimitive(Primitive prim):m_prim(prim){}

  HD_DECL
  float operator()(const float3 &x) const
  {
    return m_prim.signed_distance(x);
  }

  Primitive m_prim;
};

template<typename Primitive>
static HD_DECL
  SignedDistanceToPrimitive<Primitive> make_signed_distance_to_primitive(Primitive prim)
{
  return SignedDistanceToPrimitive<Primitive>(prim);
}


template<typename Primitive>
static HD_DECL
void extend_aabb(const Primitive &prim, 
                 optix::Aabb &box)
{
  return prim.extend_aabb(box);
}


static HD_DECL
void extend_aabb(const optix::Aabb &prim,
                 optix::Aabb &box)
{
  box.include(prim);
}



// The union of two primitives
template<typename Primitive1, typename Primitive2>
  class PrimitiveUnion
{
  public:
    // null constructor creates an undefined DistanceUnion
    HD_DECL
    PrimitiveUnion(void){}

    HD_DECL
    PrimitiveUnion(Primitive1 p1, Primitive2 p2):m_prim1(p1),m_prim2(p2){}

    HD_DECL
    float distance(const float3 &x) const
    {
      return fminf(m_prim1.distance(x), m_prim2.distance(x));
    }

    HD_DECL
    float signed_distance(const float3 &x) const
    {
      return fminf(m_prim1.signed_distance(x), m_prim2.signed_distance(x));
    }

    HD_DECL
    void extend_aabb(optix::Aabb &b) const
    {
      ::extend_aabb(m_prim1,b);
      ::extend_aabb(m_prim2,b);
    }

  protected:
    Primitive1 m_prim1;
    Primitive2 m_prim2;
};

template<typename Primitive1, typename Primitive2>
static HD_DECL
  PrimitiveUnion<Primitive1, Primitive2> make_primitive_union(Primitive1 p1, Primitive2 p2)
{
  return PrimitiveUnion<Primitive1,Primitive2>(p1,p2);
}

