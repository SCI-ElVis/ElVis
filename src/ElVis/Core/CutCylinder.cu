///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_CYLINDER_CU
#define ELVIS_CYLINDER_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "CutSurfacePayloads.cu"
#include "util.cu"
#include <ElVis/Core/OptixVariables.cu>


// Intersection program for cannonical cylinder of height 1 and centered 
// at the origin.  We expect users to provide transformation nodes 
// to rotate and resize the cylinder as needed.
RT_PROGRAM void CylinderIntersect( int primIdx )
{
  ElVisFloat3 d = MakeFloat3(ray.direction);
  ElVisFloat3 o = MakeFloat3(ray.origin);

  ElVisFloat A = d.x*d.x + d.y*d.y;
  ElVisFloat B = MAKE_FLOAT(2.0)*(o.x*d.x + o.y*d.y);
  ElVisFloat C = o.x*o.x + o.y*o.y - MAKE_FLOAT(1.0);

  ElVisFloat D = B*B - MAKE_FLOAT(4.0)*A*C;

  if( D < MAKE_FLOAT(0.0) )
  {
    return;
  }

  // In this case we know that there is at least 1 intersection.
  ElVisFloat denom = MAKE_FLOAT(2.0) * A;
  ElVisFloat square_D = Sqrtf(D);

  // Of the two roots, this is the one which is closest to the viewer.
  ElVisFloat t1 = (-B - square_D)/denom;

  if( t1 > MAKE_FLOAT(0.0) )
  {
    const ElVisFloat3 intersectionPoint = o + t1 * d;

    if( intersectionPoint.z >= MAKE_FLOAT(0.0) && intersectionPoint.z <= MAKE_FLOAT(1.0) )
    {
      if(  rtPotentialIntersection( t1 ) ) 
      {
        normal = MakeFloat3(intersectionPoint.x, intersectionPoint.y, MAKE_FLOAT(0.0));
        normalize(normal);
        rtReportIntersection(0);
      }
    }
  }

    // TODO - Uncommenting the rest of this methods causes failure.  On 1/18/2011 I postponed this so I could finish 
    // some timing tests, but it needs to be addressed.
  ElVisFloat t2 = (-B + square_D)/denom;

  if( t2 > MAKE_FLOAT(0.0) )
  {
    const ElVisFloat3 intersectionPoint = o + t2 * d;
    if( intersectionPoint.z >= MAKE_FLOAT(0.0) && intersectionPoint.z <= MAKE_FLOAT(1.0) )
    {
      if(  rtPotentialIntersection( t2 ) ) 
      {    
        // Uncomment the following line for error in Cuda 3.0 and Optix 2.0 and sm_20
        // Cuda 3.0 Optix 2.0 sm_20 - x
        // Cuda 3.0 Optix 2.0 sm_13 - x
        // Cuda 3.0 Optix 2.1 sm_20 - Works
        // Cuda 3.0 Optix 2.1 sm_13
        // Cuda 3.1 Optix 2.1 sm_20
        // Cuda 3.1 Optix 2.1 sm_13
        // Cuda 3.2 Optix 2.1 sm_20
        // Cuda 3.2 Optix 2.1 sm_13

        // 
        normal = MakeFloat3(intersectionPoint.x, intersectionPoint.y, MAKE_FLOAT(0.0));
        normalize(normal);
        rtReportIntersection(0);
      }
    }
  }

  ElVisFloat4 cap0 = MakeFloat4(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(-1.0), MAKE_FLOAT(0.0));
  ElVisFloat t3;
  if( FindPlaneIntersection(o, d, cap0, t3) )
  {
    const ElVisFloat3 intersectionPoint = o + t3 * d;
    if( intersectionPoint.x*intersectionPoint.x +
      intersectionPoint.y*intersectionPoint.y <= MAKE_FLOAT(1.0) )
    {
      if(  rtPotentialIntersection( t3 ) ) 
      {    
        normal = MakeFloat3(cap0.x, cap0.y, cap0.z);
        rtReportIntersection(0);
      }
    }
  }

  ElVisFloat4 cap1 = MakeFloat4(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0), MAKE_FLOAT(-1.0) );
  ElVisFloat t4;
  if( FindPlaneIntersection(o, d, cap1, t4) )
  {
    const ElVisFloat3 intersectionPoint = o + t4 * d;
    if( intersectionPoint.x*intersectionPoint.x +
      intersectionPoint.y*intersectionPoint.y <= MAKE_FLOAT(1.0) )
    {
      if(  rtPotentialIntersection( t4 ) ) 
      {   
        normal = MakeFloat3(cap1.x, cap1.y, cap1.z);
        rtReportIntersection(0);
      }
    }
  }

}

// Bounding box for the cannonical cylinder.
RT_PROGRAM void CylinderBounding (int, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = make_float3(-1.0f, -1.0f, -1.0f);
  aabb->m_max = make_float3(1.0f, 1.0f, 1.0f);
}


#endif
