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

#ifndef ELVIS_FIND_ELEMENT_CU
#define ELVIS_FIND_ELEMENT_CU

#include <ElVis/Core/Float.cu>
#include "CutSurfacePayloads.cu"


/// \brief Finds the element enclosing point p
__device__ __forceinline__ ElementFinderPayload FindElement(const ElVisFloat3& p)
{
    // Random direction since rays in any direction should intersect the element.  We did have some accuracy
    // issues with a direciton of (1, 0, 0) with axis-aligned elements.  This direction won't solve that
    // problem, but it does make it less likely to occur.
    const float3 direction = normalize(make_float3(1.0f, 2.0f, 3.0f));

    optix::Ray findElementRay = optix::make_Ray( ConvertToFloat3(p), direction, 1, 0.0f, RT_DEFAULT_MAX );
    ElementFinderPayload findElementPayload;
    findElementPayload.Initialize(p);

    rtTrace( element_group, findElementRay, findElementPayload);
    return findElementPayload;
}

// In this version, we don't know the reference space coordinates for the face intersection, so
// we rely on the simulation package to do the calculation for us.
__device__ ElVis::ElementId FindElement(const ElVisFloat3& testPoint, const ElVisFloat3& pointOnFace, int faceId, ElVisFloat3& faceNormal)
{
    GetFaceNormal(pointOnFace, faceId, faceNormal);
    ElVisFloat3 vectorToPointOnFace = testPoint - pointOnFace;

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

//    ELVIS_PRINTF("FindElement: Face Id %d, Normal (%f, %f, %f), test point (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", faceId,
//                 faceNormal.x, faceNormal.y, faceNormal.z,
//                 testPoint.x, testPoint.y, testPoint.z,
//                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceIdBuffer[faceId].CommonElements[0].Id,
//                 FaceIdBuffer[faceId].CommonElements[0].Type,
//                 FaceIdBuffer[faceId].CommonElements[1].Id,
//                 FaceIdBuffer[faceId].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    if( d >= 0 )
    {
        return FaceIdBuffer[faceId].CommonElements[0];
    }
    else
    {
        return FaceIdBuffer[faceId].CommonElements[1];
    }
}

__device__ ElVis::ElementId FindElement(const ElVisFloat3& testPoint, const ElVisFloat3& pointOnFace, const ElVisFloat2& referencePointOnFace, int faceId, ElVisFloat3& faceNormal)
{
    GetFaceNormal(referencePointOnFace, pointOnFace, faceId, faceNormal);
    ElVisFloat3 vectorToPointOnFace = testPoint - pointOnFace;

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

//    ELVIS_PRINTF("FindElement: Face Id %d, Normal (%f, %f, %f), test point (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", faceId,
//                 faceNormal.x, faceNormal.y, faceNormal.z,
//                 testPoint.x, testPoint.y, testPoint.z,
//                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceIdBuffer[faceId].CommonElements[0].Id,
//                 FaceIdBuffer[faceId].CommonElements[0].Type,
//                 FaceIdBuffer[faceId].CommonElements[1].Id,
//                 FaceIdBuffer[faceId].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    if( d >= 0 )
    {
        return FaceIdBuffer[faceId].CommonElements[0];
    }
    else
    {
        return FaceIdBuffer[faceId].CommonElements[1];
    }
}

#endif
