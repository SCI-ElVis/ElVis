
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
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(uint2,      launch_index, rtLaunchIndex, );
rtDeclareVariable(optix::Ray, current_ray,  rtCurrentRay,  );
rtDeclareVariable(rtObject,   dummy_object,               ,  );

rtBuffer<float4, 2> result_buffer;

// user-specified exceptions
#define MY_EXCEPTION_0    RT_EXCEPTION_USER + 0
#define MY_EXCEPTION_1    RT_EXCEPTION_USER + 1


RT_PROGRAM void buggy_draw_solid_color()
{
  // cause a stack overflow exception to occur at launch indices
  // between (0,0) & (10, 10).  (yellow)
  if( launch_index.x < 10 &&
      launch_index.y < 10 ) {
    unsigned int dummy_payload;

    optix::Ray ray = optix::make_Ray(make_float3(0,0,0), make_float3(0,0,0), 0, 0, RT_DEFAULT_MAX);
    rtTrace(dummy_object, ray, dummy_payload);
  }

  // cause a buffer index out of bounds exception to occur at launch indices
  // between (10,0) & (20, 10).  (red)
  if( 10 <= launch_index.x && launch_index.x < 20 &&
      launch_index.y < 10 ) {
    float4 x = result_buffer[make_uint2(result_buffer.size().x,result_buffer.size().y)];
  }

  // cause an invalid ray exception to occur at launch indices
  // bewteen (20,0) & (30,10).  (magenta)
  if( 20 <= launch_index.x && launch_index.x < 30 &&
      launch_index.y < 10 ) {
    unsigned int dummy_payload;

    // put a NaN in the ray direction
    float3 ray_dir = make_float3(0, 0, 0);
    ray_dir.x /= ray_dir.x; // 0.0f/0.0f produces NaN

    optix::Ray ray = optix::make_Ray(make_float3(0,0,0), ray_dir, 0, 0, RT_DEFAULT_MAX);
    rtTrace(dummy_object, ray, dummy_payload);
  }

  // throw one of two user exceptions, depending on the exact launch index
  if( 30 <= launch_index.x && launch_index.x < 40 &&
      launch_index.y < 10 ) {
    if( (launch_index.x/2+launch_index.y/2) & 1 != 0 )
      rtThrow( MY_EXCEPTION_0 );
    else
      rtThrow( MY_EXCEPTION_1 );
  }

  // cause a child index out of bounds exception to occur at launch indices
  // bewteen (40,0) & (50,10). (grey)
  if( 40 <= launch_index.x && launch_index.x < 50 &&
      launch_index.y < 10 ) {
    unsigned int dummy_payload;

    optix::Ray ray = optix::make_Ray(make_float3(0,0,0), make_float3(0,0,0), 0, 0, RT_DEFAULT_MAX);
    rtTrace(dummy_object, ray, dummy_payload);
  }

  // cause a material index out of bounds exception to occur at launch indices
  // bewteen (50,0) & (60,10). (grey)
  if( 50 <= launch_index.x && launch_index.x < 60 &&      
      launch_index.y < 10 ) {
    unsigned int dummy_payload;

    optix::Ray ray = optix::make_Ray(make_float3(0,0,0), make_float3(1,0,0), 0, 0, RT_DEFAULT_MAX);
    rtTrace(dummy_object, ray, dummy_payload);
  }

  float3 draw_color = make_float3(0.462f, 0.725f, 0.0f);
  result_buffer[launch_index] = make_float4(draw_color, 0.f);
}


RT_PROGRAM void non_terminating_miss_program()
{
  // cause infinite recursion
  unsigned int dummy_payload;
  rtTrace(dummy_object, current_ray, dummy_payload);
}

RT_PROGRAM void visit()
{
  // cause a child index out of bounds exception to occur at launch indices
  // bewteen (40,0) & (50,10). (grey)
  if( 40 <= launch_index.x && launch_index.x < 50 &&
      launch_index.y < 10 ) {
    rtIntersectChild( 1 );
  } else {
    rtIntersectChild( 0 );
  }
}

RT_PROGRAM void intersect(int)
{
  rtPotentialIntersection( 0.0f );

  // cause a material index out of bounds exception to occur at launch indices
  // bewteen (50,0) & (60,10). (grey)
  if( 50 <= launch_index.x && launch_index.x < 60 &&      
    launch_index.y < 10 ) {
    rtReportIntersection(0+1);
  }
}

RT_PROGRAM void bounds(int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->set(make_float3(-1.0f), make_float3(1.0f));
}

RT_PROGRAM void any_hit()
{
  rtTerminateRay();
}

RT_PROGRAM void closest_hit()
{

}

RT_PROGRAM void exception()
{
  // print information on the exception if printing is enabled for the context
  // (only for the first launch_index row, in order to limit the output)
  if( launch_index.y == 0 )
    rtPrintExceptionDetails();

  // mark our launch index's location in the output buffer
  // with a unique color associated with each kind of exception
  const float3 index_out_of_bounds_color        = make_float3(0.5,0.5,0.5); // grey
  const float3 buffer_index_out_of_bounds_color = make_float3(1,0,0); // red
  const float3 stack_overflow_color             = make_float3(1,1,0); // yellow
  const float3 invalid_ray_color                = make_float3(1,0,1); // magenta
  const float3 user0_color                      = make_float3(1,1,1); // white
  const float3 user1_color                      = make_float3(0,0.6f,0.85f); // blue

  float3 result;

  const unsigned int code = rtGetExceptionCode();
  switch(code) {
    case RT_EXCEPTION_INDEX_OUT_OF_BOUNDS:
      result = index_out_of_bounds_color;
      break;

    case RT_EXCEPTION_STACK_OVERFLOW:
      result = stack_overflow_color;
      break;

    case RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS:
      result = buffer_index_out_of_bounds_color;
      break;

    case RT_EXCEPTION_INVALID_RAY:
      result = invalid_ray_color;
      break;

    case MY_EXCEPTION_0:
      result = user0_color;
      break;

    case MY_EXCEPTION_1:
      result = user1_color;
      break;

    default:
      result = make_float3(0,0,0); // black for unhandled exceptions
      break;
  }

  result_buffer[launch_index] = make_float4(result, 0.f);
}

