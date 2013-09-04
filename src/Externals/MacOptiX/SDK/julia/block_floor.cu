
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
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
using namespace optix;
#include "distance_field.h"

const size_t NUM_REPETITIONS = 320;

#define HD_DECL_CLASS HD_DECL __forceinline__
#define HD_DECL_FUNC  static HD_DECL

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

struct PerRayData_shadow
{
  float3 attenuation;
};

rtDeclareVariable(float, floor_time, , );
rtDeclareVariable(float4, sphere, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, isect_t, rtIntersectionDistance, );
rtDeclareVariable(float3, normal, attribute normal, ); 
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(rtObject,                         top_shadower, , );
rtDeclareVariable(unsigned int,                     shadowsActive, , );



__constant__ float u01[] = {  1.0f / 64, 31.0f / 64,  6.0f / 64, 60.0f / 64, 38.0f / 64, 12.0f / 64, 13.0f / 64, 56.0f / 64,
                             25.0f / 64, 59.0f / 64, 21.0f / 64, 62.0f / 64, 32.0f / 64, 44.0f / 64, 11.0f / 64, 49.0f / 64,
                             41.0f / 64, 52.0f / 64, 35.0f / 64, 45.0f / 64, 34.0f / 64, 36.0f / 64, 33.0f / 64, 15.0f / 64,
                             57.0f / 64,  3.0f / 64, 19.0f / 64,  7.0f / 64, 14.0f / 64, 47.0f / 64, 17.0f / 64,  0.0f / 64,
                              2.0f / 64, 16.0f / 64,  4.0f / 64, 46.0f / 64, 43.0f / 64, 10.0f / 64, 20.0f / 64, 24.0f / 64,
                             29.0f / 64, 42.0f / 64, 54.0f / 64, 50.0f / 64, 18.0f / 64, 58.0f / 64, 51.0f / 64, 27.0f / 64,
                             37.0f / 64, 22.0f / 64, 39.0f / 64, 28.0f / 64, 40.0f / 64,  8.0f / 64, 26.0f / 64, 55.0f / 64,
                             63.0f / 64,  5.0f / 64, 48.0f / 64, 61.0f / 64,  9.0f / 64, 53.0f / 64, 30.0f / 64, 23.0f / 64};

// XXX this buggy code causes some really interesting patterns
//HD_DECL_FUNC
//float hash(unsigned int x)
//{
//  x = (x ^ 61) ^ (x >> 16);
//  x = x + (x << 3);
//  x = x ^ (x >> 4);
//  x = x * 0x27d4eb2d;
//  x = x ^ (x >> 15);
//  return x;
//}
//
//HD_DECL
//float rand01(unsigned int x, unsigned int y)
//{
//  //return static_cast<float>(hash(hash(x) + hash(y))) / UINT_MAX;
//  //return static_cast<float>(hash(hash(x) + y)) / UINT_MAX;
//}

HD_DECL_FUNC
unsigned int hash(unsigned int x)
{
  x = (x ^ 61) ^ (x >> 16);
  x = x + (x << 3);
  x = x ^ (x >> 4);
  x = x * 0x27d4eb2d;
  x = x ^ (x >> 15);
  return x;
}

// XXX couldn't get this to produce anything useful
#if 0
HD_DECL_FUNC
int hash64(long key)
{
  key = (~key) + (key << 18); // key = (key << 18) - key - 1;
  key = key ^ (key >> 31);
  key = key * 21; // key = (key + (key << 2)) + (key << 4);
  key = key ^ (key >> 11);
  key = key + (key << 6);
  key = key ^ (key >> 22);
  return (int)key;
}
#endif

HD_DECL_FUNC
float rand01(unsigned int x, unsigned int y)
{
  return static_cast<float>(hash(hash(x) + y)) / UINT_MAX;
}

HD_DECL_FUNC
void divide(float numerator, float denominator,
            float &whole, float &remainder)
{
  whole = floorf(numerator / denominator);
  remainder = (numerator - whole*denominator) + (numerator < 0 ? denominator : 0);
}



template<typename Primitive, bool include_conservative_tests>
  struct PrimitiveRepeatX
{
  HD_DECL_CLASS
  PrimitiveRepeatX(void){}

  HD_DECL_CLASS
  PrimitiveRepeatX(Primitive prim, unsigned int num_repetitions, const float epsilon)
    :m_num_repetitions(num_repetitions),m_aabb(),m_prim(prim),m_epsilon(epsilon)
  {
    // figure out the bounding box
    ::extend_aabb(m_prim, m_aabb);

    m_origin = m_aabb.m_min.x - epsilon;
    m_period = m_aabb.extent().x + 2.0f * m_epsilon;

    m_aabb.m_min.x = m_origin;
    m_aabb.m_max.x = m_aabb.m_min.x + static_cast<float>(m_num_repetitions) * m_period;
  }

  template<bool use_signed_distance>
  HD_DECL_CLASS
  float distance_to_instanced_primitive(const float3 &x,
                                        unsigned int index_x,
                                        unsigned int index_z,
                                        const float bias_x) const
  {
    // transform the primitive
    float n = rand01(index_x, index_z);
    n *= 0.5f * (__sinf(floor_time + index_x) + 1.0f);

    // transform the primitive
    float height = n * m_prim.extent().y;
    Primitive shortened_prim = m_prim; 
    shortened_prim.m_max.y = shortened_prim.m_min.y + height;

    shortened_prim.m_min.x += bias_x;
    shortened_prim.m_max.x += bias_x;

    float result;

    if(use_signed_distance)
      result = shortened_prim.signedDistance(x);
    else
      result = shortened_prim.distance(x);
    return result;
  }

  template<bool use_signed_distance>
  HD_DECL_CLASS
  float distance(float3 x, const unsigned int index_z) const
  {
    float result;

    float index_x = 0;

    // is x to the right of the bounding box?
    if(x.x > m_aabb.m_max.x)
    {
      x.x -= (m_aabb.m_max.x - m_period - m_origin);
      index_x = m_num_repetitions - 1;
    }
    // is x within the bounding box?
    else if(x.x >= m_aabb.m_min.x)
    {
      // subtract out the origin
      x.x -= m_origin;

      // find the whole part & fractional part of this division
      divide(x.x, m_period, index_x, x.x);

      // add the origin back
      x.x += m_origin;
    }

    result = distance_to_instanced_primitive<use_signed_distance>(x, index_x, index_z, 0);

    if(include_conservative_tests) {
      // we have to include the distance to the neighboring instances here
      // as well, or we may miss them when we exit this cell
      if(0 < index_x)
        result = fminf(result, distance_to_instanced_primitive<use_signed_distance>(x, index_x - 1, index_z, -m_period));

      if(index_x < m_num_repetitions - 1)
        result = fminf(result, distance_to_instanced_primitive<use_signed_distance>(x, index_x + 1, index_z,  m_period));
    } else {
      float distance_to_right_wall = (m_origin + m_period) - x.x;
      float distance_to_left_wall  = x.x - m_origin;

      distance_to_left_wall += m_epsilon;
      distance_to_right_wall += m_epsilon;

      result = fminf(result, fminf(distance_to_left_wall, distance_to_right_wall));
    }

    return result;
  }

  HD_DECL_CLASS
  float distance(const float3 &x, const unsigned int index_z) const
  {
    return distance<false>(x,index_z);
  }

  HD_DECL_CLASS
  float distance(const float3 &x) const
  {
    return distance(x, 0);
  }

  HD_DECL_CLASS
  float signed_distance(const float3 &x, const unsigned int index_z) const
  {
    return distance<true>(x,index_z);
  }

  HD_DECL_CLASS
  float signed_distance(const float3 &x) const
  {
    return signed_distance(x,0);
  }

  HD_DECL_CLASS
  void extend_aabb(optix::Aabb &b) const
  {
    b.include(m_aabb);
  }

  unsigned int m_num_repetitions;
  optix::Aabb m_aabb;
  Primitive m_prim;
  float m_origin;
  float m_period;
  float m_epsilon;
};


template<bool include_conservative_tests, typename Primitive>
HD_DECL_FUNC
  PrimitiveRepeatX<Primitive, include_conservative_tests>
    make_primitive_repeat_x(Primitive prim, size_t n)
{
  optix::Aabb b;
  extend_aabb(prim, b);

  return PrimitiveRepeatX<Primitive, include_conservative_tests>(prim, n, 0.025f);
}



template<typename Primitive, bool include_conservative_tests>
  struct PrimitiveRepeatZ
{
  HD_DECL_CLASS
  PrimitiveRepeatZ(void){}

  HD_DECL_CLASS
  PrimitiveRepeatZ(Primitive prim, unsigned int num_repetitions, const float epsilon)
    :m_num_repetitions(num_repetitions),m_aabb(),m_prim(prim),m_epsilon(epsilon)
  {
    // figure out the bounding box
    ::extend_aabb(m_prim, m_aabb);

    m_origin = m_aabb.m_min.z - epsilon;
    m_period = m_aabb.extent().z + 2.0f * m_epsilon;

    m_aabb.m_min.z = m_origin;
    m_aabb.m_max.z = m_aabb.m_min.z + static_cast<float>(m_num_repetitions) * m_period;
  }

  template<bool use_signed_distance>
  HD_DECL_CLASS
  float distance_to_instanced_primitive(const float3 &x,
                                        const unsigned int index_z,
                                        const float bias_z) const
  {
    Primitive translated_prim = m_prim;

    // translate the Primitive
    translated_prim.m_prim.m_min.z += bias_z;
    translated_prim.m_prim.m_max.z += bias_z;

    float result;
    if(use_signed_distance) {
      result = translated_prim.signed_distance(x, index_z);
    } else
    {
      result = translated_prim.distance(x, index_z);
    }

    return result;
  }

  template<bool use_signed_distance>
  HD_DECL_CLASS
  float distance(float3 x) const
  {
    float result;

    float index_z = 0;

    // is z to the right of the bounding box?
    if(x.z > m_aabb.m_max.z)
    {
      x.z -= (m_aabb.m_max.z - m_period - m_origin);
      index_z = m_num_repetitions - 1;
    }
    // is z within the bounding box?
    else if(x.z >= m_aabb.m_min.z)
    {
      // subtract out the origin
      x.z -= m_origin;

      // find the whole part & fractional part of this division
      divide(x.z, m_period, index_z, x.z);

      // add the origin back
      x.z += m_origin;
    }

    result = distance_to_instanced_primitive<use_signed_distance>(x, index_z, 0);

    if(include_conservative_tests) {
      // check left?
      if(index_z > 0)
        result = fminf(result, distance_to_instanced_primitive<use_signed_distance>(x, index_z - 1, -m_period));

      // check right?
      if(index_z < m_num_repetitions - 1)
        result = fminf(result, distance_to_instanced_primitive<use_signed_distance>(x, index_z + 1,  m_period));
    } else {
      float distance_to_right_wall = (m_origin + m_period) - x.z;
      float distance_to_left_wall  = x.z - m_origin;

      distance_to_left_wall += m_epsilon;
      distance_to_right_wall += m_epsilon;

      result = fminf(result, fminf(distance_to_left_wall, distance_to_right_wall));
    }

    return result;
  }

  HD_DECL_CLASS
  float distance(const float3 &x) const
  {
    return distance<false>(x);
  }

  HD_DECL_CLASS
  float signed_distance(const float3 &x) const
  {
    return distance<true>(x);
  }

  HD_DECL_CLASS
  void extend_aabb(optix::Aabb &b) const
  {
    b.include(m_aabb);
  }

  unsigned int m_num_repetitions;
  optix::Aabb m_aabb;
  Primitive m_prim;
  float m_origin;
  float m_period;
  float m_epsilon;
};


template<bool include_conservative_tests, typename Primitive>
HD_DECL_FUNC
  PrimitiveRepeatZ<Primitive, include_conservative_tests>
    make_primitive_repeat_z(Primitive prim, size_t n)
{
  optix::Aabb b;
  extend_aabb(prim, b);

  return PrimitiveRepeatZ<Primitive, include_conservative_tests>(prim, n, 0.025f);
}


template<typename Primitive>
  struct TwistY
{
  HD_DECL_CLASS
  TwistY(void){}

  HD_DECL_CLASS
  TwistY(Primitive prim):m_prim(prim){}

  HD_DECL_CLASS
  float3 transform(const float3 &x) const
  {
    optix::Aabb b;
    ::extend_aabb(m_prim, b);
    float3 o = b.center();

    float3 x2 = x;

    float3 temp = x2;

    const float theta = temp.y;
    x2.x =  cosf(theta)*temp.x + sinf(theta)*temp.z;
    x2.z = -sinf(theta)*temp.x + cosf(theta)*temp.z;

    x2.x -= b.m_min.x;
    x2.z -= b.m_min.z;

    return x2;
  }

  HD_DECL_CLASS
  float distance(float3 x) const
  {
    x = transform(x);
    return m_prim.distance(x);
  }

  HD_DECL_CLASS
  float signed_distance(float3 x) const
  {
    x = transform(x);
    return m_prim.signed_distance(x);
  }

  HD_DECL_CLASS
  void extend_aabb(optix::Aabb &b) const
  {
    ::extend_aabb(m_prim, b);
    b.enlarge(0.1f);
  }

  Primitive m_prim;
};


struct Sphere
{
  HD_DECL_CLASS
  float signed_distance(const float3 &x) const
  {
    return length(x - make_float3(sphere)) - sphere.w;
  }

  HD_DECL_CLASS
  float distance(const float3 &x) const
  {
    return fmaxf(signed_distance(x), 0.0f);
  }
  float placeholder;
  HD_DECL_CLASS Sphere() : placeholder(0) {}
};



template<bool use_conservative_tests>
  struct object_factory
{
  typedef optix::Aabb Block;
  typedef TwistY<Block> Screw;
  typedef Block Primitive;
  typedef PrimitiveRepeatX<Primitive, use_conservative_tests> RowOfBlocks;
  typedef PrimitiveRepeatZ<RowOfBlocks, use_conservative_tests> Floor;
  typedef Floor Object;
  
  static HD_DECL_CLASS
  void make_object(Object &x, const float3 &ray_direction)
  {
    optix::Aabb bounds_of_floor( make_float3(-100.0f, -2.0f, -100.0f),
                                       make_float3( 100.0f, -1.0f,  100.0f));
  
    float3 block_extent;
    block_extent.x = bounds_of_floor.extent().x / NUM_REPETITIONS;
    block_extent.y = bounds_of_floor.extent().y;
    block_extent.z = bounds_of_floor.extent().z / NUM_REPETITIONS;
    Primitive prim = Primitive( Block(bounds_of_floor.m_min,
                                      bounds_of_floor.m_min + block_extent));
  
    RowOfBlocks row_of_prims = make_primitive_repeat_x<use_conservative_tests>(prim, NUM_REPETITIONS);

    Floor floor = make_primitive_repeat_z<use_conservative_tests>(row_of_prims, NUM_REPETITIONS);
  
    x = floor;
  }
};


template<typename Primitive>
HD_DECL_FUNC
bool intersect_aabb(optix::Ray &r, Primitive prim)
{
  optix::Aabb b;
  prim.extend_aabb(b);

  float3 t0 = (b.m_min - r.origin)/r.direction;
  float3 t1 = (b.m_max - r.origin)/r.direction;
  float3 near = fminf(t0, t1);
  float3 far = fmaxf(t0, t1);
  float tmin = fmaxf( near );
  float tmax = fminf( far );

  if(tmin <= tmax && tmin <= r.tmax) {
    r.tmin = max(r.tmin,tmin);
    return true;
  }

  return false;
}


RT_PROGRAM void intersect(int primIdx)
{
  object_factory<false>::Object obj;
  object_factory<false>::make_object(obj, ray.direction);

  // first check for intersection between the ray and aabb
  optix::Ray tmp_ray = ray;
  if(intersect_aabb(tmp_ray, obj)) {
    float epsilon = 1.25e-3f;
    float max_epsilon = 2.5e-2f;

    float3 hit_point;
    float t = adaptive_sphere_trace<1000>(tmp_ray, make_distance_to_primitive(obj), hit_point, epsilon, max_epsilon);
    if(t < tmp_ray.tmax)
    {
      if(rtPotentialIntersection(t))
      {
        // we need to use signed distance when estimating the normal
        normal = estimate_normal(make_signed_distance_to_primitive(obj), hit_point, 1e-5f);
        rtReportIntersection(0);
      }
    }
  }
}

RT_PROGRAM void bounds (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;

  object_factory<false>::Object x;
  object_factory<false>::make_object(x, make_float3(0,0,0));

  // start with empty Aabb
  *aabb = optix::Aabb();

  // extend to x's Aabb
  x.extend_aabb(*aabb);
}


RT_PROGRAM void block_floor_ch_radiance(void)
{
  const float3 p = ray.origin + isect_t * ray.direction;

  // ambient occlusion - use the conservative distance tests which check neighboring blocks
  object_factory<true>::Object obj;
  object_factory<true>::make_object(obj, make_float3(0,0,0));
  //occlusion = magic_ambient_occlusion<5>(p, normal, 0.35f, 1.0f, make_distance_to_primitive(obj));
  float occlusion = magic_ambient_occlusion<5>(p, normal, 0.35f, 1.0f, make_distance_to_primitive(make_primitive_union(Sphere(),obj)));
  prd_radiance.result = make_float3(occlusion);

  if ( shadowsActive ) {
    PerRayData_shadow prd_shd;
    prd_shd.attenuation = make_float3(1.0f);
    optix::Ray shdray = optix::make_Ray( p, normalize(make_float3(0,1,0)), 1, 0.2f, RT_DEFAULT_MAX );
    rtTrace( top_shadower, shdray, prd_shd );
    prd_radiance.result *= 1.0f - 0.3f * (1.0f-prd_shd.attenuation);
  }
}

RT_PROGRAM void block_floor_ah_shadow()
{
  rtTerminateRay();
}

