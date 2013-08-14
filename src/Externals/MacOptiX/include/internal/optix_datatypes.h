
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

#ifndef __optix_optix_datatypes_h__
#define __optix_optix_datatypes_h__

#include <host_defines.h>               /* for __inline__ */
#include "../optixu/optixu_vector_types.h"        /* for float3 */
#include "optix_declarations.h"         /* for RT_HOSTDEVICE */

#ifdef __cplusplus
namespace optix {
#endif

#define RT_DEFAULT_MAX 1.e27f

/*
   Rays
*/

struct Ray {

#ifdef __cplusplus
  __inline__ RT_HOSTDEVICE
  Ray(){}

  __inline__ RT_HOSTDEVICE
  Ray( const Ray &r)
    :origin(r.origin),direction(r.direction),ray_type(r.ray_type),tmin(r.tmin),tmax(r.tmax){}

  __inline__ RT_HOSTDEVICE
  Ray( float3 origin_, float3 direction_, unsigned int ray_type_, float tmin_, float tmax_ = RT_DEFAULT_MAX )
    :origin(origin_),direction(direction_),ray_type(ray_type_),tmin(tmin_),tmax(tmax_){}
#endif

  float3 origin;
  float3 direction;
  unsigned int ray_type;
  float tmin;
  float tmax;
};

__inline__ RT_HOSTDEVICE
Ray make_Ray( float3 origin, float3 direction, unsigned int ray_type, float tmin, float tmax )
{
  Ray ray;
  ray.origin = origin;
  ray.direction = direction;
  ray.ray_type = ray_type;
  ray.tmin = tmin;
  ray.tmax = tmax;
  return ray;
}

#ifdef __cplusplus
} // namespace
#endif

#endif /* __optix_optix_datatypes_h__ */
