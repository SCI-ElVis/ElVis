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

#ifndef ELVIS_ELEMENT_TRAVERSAL_CU
#define ELVIS_ELEMENT_TRAVERSAL_CU

#include <ElVis/Core/PrimaryRayGenerator.cu>
#include <ElVis/Core/Cuda.h>
#include <ElVis/Core/FaceIntersection.cu>

rtDeclareVariable(ElVisFloat, FaceTolerance, , );

struct Segment
{
  __device__ Segment() :
    Start(MAKE_FLOAT(0.0)),
    End(MAKE_FLOAT(0.0)),
    ElementId(-1),
    ElementTypeId(-1),
    RayDirection(MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0)))
  {
  }

  ElVisFloat Start;
  ElVisFloat End;
  int ElementId;
  int ElementTypeId;
  ElVisFloat3 RayDirection;
};


__device__ bool FindNextSegmentAlongRay(Segment& seg, const ElVisFloat3& rayDirection)
{
  optix::size_t2 screen = color_buffer.size();
  ELVIS_PRINTF("FindNextSegmentAlongRay: Starting t %f \n", seg.Start);

  // If we have already encountered an object we don't need to continue along this ray.
  ElVisFloat depth = depth_buffer[launch_index];
  //ELVIS_PRINTF("FindNextSegmentAlongRay best depth so far %2.10f\n", depth);
  if( depth < seg.Start )
  {
    return false;
  }

  ElVisFloat3 origin = eye + seg.Start*rayDirection;

  VolumeRenderingPayload payload = FindNextFaceIntersection(origin, rayDirection);
  
  if( !payload.FoundIntersection )
  {
    //ELVIS_PRINTF("Did not find element intersection.\n");
    return false;
  }

  seg.End = seg.Start + payload.IntersectionT;
  ELVIS_PRINTF("Segment is [%f, %f]\n", seg.Start, seg.End);

  ElementFinderPayload newApproach = findElementFromFace(origin, rayDirection, payload);
  ELVIS_PRINTF("FindNextSegmentAlongRay: Segment element %d and type %d, New approach id %d and type %d\n",
    seg.ElementId, seg.ElementTypeId, newApproach.elementId, newApproach.elementType);
  seg.ElementId = newApproach.elementId;
  seg.ElementTypeId = newApproach.elementType;
  return true;
}

__device__ bool ValidateSegment(const Segment& seg)
{
  int elementId = seg.ElementId;
  //ELVIS_PRINTF("ValidateSegment: Element id %d\n", elementId);

  if( elementId == -1 )
  {
    //ELVIS_PRINTF("ValidateSegment: Exiting because element id is -1\n");
    return false;
  }

  int elementTypeId = seg.ElementTypeId;

  ElVisFloat a = seg.Start;
  ElVisFloat b = seg.End;

  ElVisFloat3 rayDirection = seg.RayDirection;
  ElVisFloat d = (b-a);

  //ELVIS_PRINTF("ValidateSegment: Ray Direction (%2.10f, %2.10f, %2.10f), segment distance %2.10f and endopints [%2.10f, %2.10f]\n", rayDirection.x, rayDirection.y, rayDirection.z, d, a, b);

  if( d == MAKE_FLOAT(0.0) )
  {
    //ELVIS_PRINTF("ValidateSegment: Exiting because d is 0\n", rayDirection.x, rayDirection.y, rayDirection.z, d);
    return false;
  }

  return true;
}

template<typename SegmentFunction>
__device__ void ElementTraversal(SegmentFunction& f)
{
  // Cast a single ray to find entrance to volume.
  optix::size_t2 screen = color_buffer.size();
  optix::Ray initialRay = GeneratePrimaryRay(screen, 2, 1e-3f);

  ElVisFloat3 origin0 = MakeFloat3(initialRay.origin);
  ElVisFloat3 rayDirection = MakeFloat3(initialRay.direction);

  depth_buffer[launch_index] = ELVIS_FLOAT_MAX;
  Segment seg;
  seg.RayDirection = rayDirection;
  int maxIter = 200;
  int iter = 0;
  while( FindNextSegmentAlongRay(seg, rayDirection) && iter < maxIter)
  {
    if( seg.End < MAKE_FLOAT(0.0) )
    {
      ELVIS_PRINTF("ElementTraversal: Exiting because ray has left volume based on segment end\n");
      return;
    }

    if(ValidateSegment(seg) && f(seg, origin0) )
    {
      ELVIS_PRINTF("ElementTraversal: Done because segment is valid and function indicates we are done.\n");
      return;
    }

    seg.Start = seg.End;
    ++iter;
  }
}

#endif

