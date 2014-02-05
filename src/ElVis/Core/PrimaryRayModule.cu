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

#ifndef ELVIS_CORE_PRIMARY_RAY_MODULE_CU
#define ELVIS_CORE_PRIMARY_RAY_MODULE_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "CutSurfacePayloads.cu"
#include "ConvertToColor.cu"
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/PrimaryRayGenerator.cu>
#include <ElVis/Core/OptixVariables.cu>


RT_PROGRAM void GeneratePrimaryRays()
{   
  optix::size_t2 screen = color_buffer.size();

  optix::Ray ray = GeneratePrimaryRay(screen, 0, 1e-3f);
  CutSurfaceScalarValuePayload payload;
  payload.Initialize();

  normal_buffer[launch_index] = payload.Normal;
  SampleBuffer[launch_index] = payload.scalarValue;

  ELVIS_PRINTF("Origin (%2.15f, %2.15f, %2.15f), Direction (%2.15f, %2.15f, %2.15f)\n",
    ray.origin.x, ray.origin.y, ray.origin.z,
    ray.direction.x, ray.direction.y, ray.direction.z);
  rtTrace(SurfaceGeometryGroup, ray, payload);   


  ELVIS_PRINTF("GeneratePrimaryRays: Normal (%f, %f, %f)\n", payload.Normal.x, payload.Normal.y, payload.Normal.z);
  raw_color_buffer[launch_index] = payload.Color;
  color_buffer[launch_index] = ConvertToColor(payload.Color);
  normal_buffer[launch_index] = payload.Normal;
  intersection_buffer[launch_index] = payload.IntersectionPoint;
  SampleBuffer[launch_index] = payload.scalarValue;
  ElementIdBuffer[launch_index] = payload.elementId;
  ElementTypeBuffer[launch_index] = payload.elementType;

  ////unsigned int s = 0x01 << (DepthBits-1);
  ////depth_buffer[launch_index] = (far+near)/(far-near) + 1.0f/payload.IntersectionT * ( (2.0f*far*near)/(far-near));
  ////    if( payload.IntersectionT == -1.0f )
  ////    {
  ////        // Assuming a depth buffer bound on [0,1], the normal LESS comparision for the depth buffer will ignore these
  ////        // pixels.
  ////        depth_buffer[launch_index] = 2.0f;
  ////    }
  ////    else
  ////    {
  ////        depth_buffer[launch_index] = ( 1.0f/payload.IntersectionT * far*near/(far-near) + .5f*(far+near)/(far-near) + .5f );
  ////    }
  ////    ELVIS_PRINTF("Intersection t %f and depth t %f\n", payload.IntersectionT, depth_buffer[launch_index]);

  //// Probably want window coordinates.

  //// clip coordinates
  ////    ELVIS_PRINTF("Clip Coordinates.\n");
  ////    if( payload.IntersectionT == -1.0f )
  ////    {
  ////        // Assuming a depth buffer bound on [0,1], the normal LESS comparision for the depth buffer will ignore these
  ////        // pixels.
  ////        depth_buffer[launch_index] = 2.0f;
  ////    }
  ////    else
  ////    {
  ////        depth_buffer[launch_index] = -payload.IntersectionT * (far+near)/(far-near) - 1.0f * 2.0f *far*near/(far-near);
  ////    }


  //// ndc Coordinates
  //// These are -1..1, I want 0..1
  //if( !payload.isValid )
  //{
  //  // Assuming a depth buffer bound on [0,1], the normal LESS comparision for the depth buffer will ignore these
  //  // pixels.
  //  depth_buffer[launch_index] = ELVIS_FLOAT_MAX;
  //}
  //else
  //{
  //  //depth_buffer[launch_index] = (far+near)/(far-near) - 2.0f/payload.IntersectionT * far*near/(far-near);
  //  //depth_buffer[launch_index] = (depth_buffer[launch_index]+1.0)/2.0;
  //  depth_buffer[launch_index] = payload.IntersectionT;
  //}

  //ELVIS_PRINTF("GeneratePrimaryRays: Intersection t %f and depth t %f\n", payload.IntersectionT, depth_buffer[launch_index]);
}



RT_PROGRAM void PrimaryRayMissed()
{
}

RT_PROGRAM void ExceptionProgram()
{
  rtPrintf("Handling OptiX exception.  Details:\n");
  rtPrintExceptionDetails();
}





#endif

