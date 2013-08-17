
// Copyright NVIDIA Corporation 2009
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include "helpers.h"
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

rtDeclareVariable(float,    overcast, , );
rtDeclareVariable(optix::float3,   sun_direction, , );
rtDeclareVariable(optix::float3,   sun_color, , );
rtDeclareVariable(optix::float3,   sky_up, , );

rtDeclareVariable(optix::float3, inv_divisor_Yxy, ,);
rtDeclareVariable(optix::float3, c0, ,);
rtDeclareVariable(optix::float3, c1, ,);
rtDeclareVariable(optix::float3, c2, ,);
rtDeclareVariable(optix::float3, c3, ,);
rtDeclareVariable(optix::float3, c4, ,);



static __host__ __device__ optix::float3 querySkyModel( bool CEL, const optix::float3& direction )
{
  using namespace optix;

  float3 overcast_sky_color = make_float3( 0.0f );
  float3 sunlit_sky_color   = make_float3( 0.0f );

  // Preetham skylight model
  if( overcast < 1.0f ) {
    float3 ray_direction = direction;
    if( CEL && dot( ray_direction, sun_direction ) > 94.0f / sqrtf( 94.0f*94.0f + 0.45f*0.45f) ) {
      sunlit_sky_color = sun_color;
    } else {
      float inv_dir_dot_up = 1.f / dot( ray_direction, sky_up ); 
      if(inv_dir_dot_up < 0.f) {
        ray_direction = reflect(ray_direction, sky_up );
        inv_dir_dot_up = -inv_dir_dot_up;
      }

      float gamma = dot(sun_direction, ray_direction);
      float acos_gamma = acosf(gamma);
      float3 A =  c1 * inv_dir_dot_up;
      float3 B =  c3 * acos_gamma;
      float3 color_Yxy = ( make_float3( 1.0f ) + c0*make_float3( expf( A.x ),expf( A.y ),expf( A.z ) ) ) *
        ( make_float3( 1.0f ) + c2*make_float3( expf( B.x ),expf( B.y ),expf( B.z ) ) + c4*gamma*gamma );
      color_Yxy *= inv_divisor_Yxy;

      color_Yxy.y = 0.33f + 1.2f * ( color_Yxy.y - 0.33f ); // Pump up chromaticity a bit
      color_Yxy.z = 0.33f + 1.2f * ( color_Yxy.z - 0.33f ); //
      float3 color_XYZ = Yxy2XYZ( color_Yxy );
      sunlit_sky_color = XYZ2rgb( color_XYZ ); 
      sunlit_sky_color /= 1000.0f; // We are choosing to return kilo-candellas / meter^2
    }
  }

  // CIE standard overcast sky model
  float Y =  15.0f;
  overcast_sky_color = make_float3( ( 1.0f + 2.0f * fabsf( direction.y ) ) / 3.0f * Y );

  // return linear combo of the two
  return lerp( sunlit_sky_color, overcast_sky_color, overcast );
}
