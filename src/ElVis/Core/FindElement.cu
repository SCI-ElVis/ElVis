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
#include <ElVis/Core/FaceIntersection.cu>

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

    ELVIS_PRINTF("FindElement: Finding element enclosing point (%f, %f, %f)\n", p.x, p.y, p.z);
    rtTrace( PointLocationGroup, findElementRay, findElementPayload);
    ELVIS_PRINTF("FindElement: Found Element %d\n", findElementPayload.elementId);
    return findElementPayload;
}

__device__ __forceinline__ ElementFinderPayload findElementFromFace(const ElVisFloat3& p, const ElVisFloat3& direction, const VolumeRenderingPayload& payload_v)
{
    ElementFinderPayload findElementPayload;
    findElementPayload.Initialize(p);
    if( payload_v.FoundIntersection == 0 )
    {
        ELVIS_PRINTF("FindElementFromFace: Did not find element intersection.\n");
        return findElementPayload;
    }
    else
    {
        ELVIS_PRINTF("FindElementFromFace: Found element intersection.\n");
        findElementPayload.elementId = 0;
        findElementPayload.elementType = 0;
    }

    ElVisFloat3 faceNormal;
    ElVisFloat3 pointOnFace = p + payload_v.IntersectionT*direction;
    GetFaceNormal(pointOnFace, payload_v.FaceId, faceNormal);

    ElVisFloat3 vectorToPointOnFace = p - pointOnFace;

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

    ELVIS_PRINTF("FindElementFromFace: Face Id %d, Normal (%f, %f, %f), point on face (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", payload_v.FaceId,
                 faceNormal.x, faceNormal.y, faceNormal.z,
                 pointOnFace.x, pointOnFace.y, pointOnFace.z,
                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
    ELVIS_PRINTF("FindElementFromFace: Face buffer size: %d\n", FaceIdBuffer.size());
     
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceIdBuffer[faceId].CommonElements[0].Id,
//                 FaceIdBuffer[faceId].CommonElements[0].Type,
//                 FaceIdBuffer[faceId].CommonElements[1].Id,
//                 FaceIdBuffer[faceId].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    ElVis::ElementId id;
    if( d >= 0 )
    {
        
        id = FaceIdBuffer[payload_v.FaceId].CommonElements[0];
    }
    else
    {
        id = FaceIdBuffer[payload_v.FaceId].CommonElements[1];
    }
    
    findElementPayload.elementId = id.Id;
    findElementPayload.elementType = id.Type;
    //findElementPayload.IntersectionPoint = pointOnFace;
    ELVIS_PRINTF("FindElementFromFace: Element Id %d and Type %d\n", id.Id, id.Type);
    ELVIS_PRINTF("FindElementFromFace: Inside id %d and tpye %d, outside id %d and type %d\n", 
        FaceIdBuffer[payload_v.FaceId].CommonElements[0].Id,
        FaceIdBuffer[payload_v.FaceId].CommonElements[0].Type,
        FaceIdBuffer[payload_v.FaceId].CommonElements[1].Id,
        FaceIdBuffer[payload_v.FaceId].CommonElements[1].Type);
    return findElementPayload;
}

/// \brief Finds the element enclosing point p
__device__ __forceinline__ ElementFinderPayload FindElementFromFace(const ElVisFloat3& p)
{
    // Random direction since rays in any direction should intersect the element.  We did have some accuracy
    // issues with a direciton of (1, 0, 0) with axis-aligned elements.  This direction won't solve that
    // problem, but it does make it less likely to occur.
    ElVisFloat3 direction = MakeFloat3(ray.direction);

    ELVIS_PRINTF("FindElementFromFace: Looking for element that encloses point (%f, %f, %f)\n", p.x, p.y, p.z);

    VolumeRenderingPayload payload_v = FindNextFaceIntersection(p, direction);
   
    ELVIS_PRINTF("FindElementFromFace: First 1 Found %d T %f id %d\n", payload_v.FoundIntersection,
        payload_v.IntersectionT, payload_v.FaceId);

    ElementFinderPayload findElementPayload = findElementFromFace(p, direction, payload_v);
    if( payload_v.FoundIntersection == 1 && findElementPayload.elementId == -1 )
    {
        payload_v.Initialize();
        direction = MakeFloat3(-direction.x, -direction.y, -direction.z);
        payload_v = FindNextFaceIntersection(p, direction);
        ELVIS_PRINTF("FindElementFromFace Try 2: Found %d T %f id %d\n", payload_v.FoundIntersection,
            payload_v.IntersectionT, payload_v.FaceId);
        findElementPayload = findElementFromFace(p, direction, payload_v);
    }

    return findElementPayload;
    //findElementPayload.Initialize(p);
    //if( payload_v.FoundIntersection == 0 )
    //{
    //    ELVIS_PRINTF("FindElementFromFace: Did not find element intersection.\n");
    //    return findElementPayload;
    //}
    //else
    //{
    //    ELVIS_PRINTF("FindElementFromFace: Found element intersection.\n");
    //    findElementPayload.elementId = 0;
    //    findElementPayload.elementType = 0;
    //}

    //findElementPayload = findElementFromFace(p, direction, payload_v);

    //ELVIS_PRINTF("FindElementFromFace: Returning findElementPayload.\n");
    //ELVIS_PRINTF("FindElementFromFace: IntersectionPoint (%f, %f, %f).\n", findElementPayload.IntersectionPoint.x, findElementPayload.IntersectionPoint.y, findElementPayload.IntersectionPoint.z);
    //ELVIS_PRINTF("FindElementFromFace: Element id = %d and element type = %d.\n", findElementPayload.elementId, findElementPayload.elementType);
    //ELVIS_PRINTF("FindElementFromFace: ReferencePointType = %d.\n", findElementPayload.ReferencePointType);
    //ELVIS_PRINTF("FindElementFromFace: ReferenceIntersectionPoint = (%f, %f, %f).\n", findElementPayload.ReferenceIntersectionPoint.x, findElementPayload.ReferenceIntersectionPoint.y, findElementPayload.ReferenceIntersectionPoint.z);


    return findElementPayload;


//    ElVisFloat3 faceNormal;
//    ElVisFloat3 pointOnFace = p + payload_v.IntersectionT*direction;
//    GetFaceNormal(pointOnFace, payload_v.FaceId, faceNormal);
//
//    ElVisFloat3 vectorToPointOnFace = p - pointOnFace;
//
//    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);
//
//    ELVIS_PRINTF("FindElementFromFace: Face Id %d, Normal (%f, %f, %f), point on face (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", payload_v.FaceId,
//                 faceNormal.x, faceNormal.y, faceNormal.z,
//                 pointOnFace.x, pointOnFace.y, pointOnFace.z,
//                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
//    ELVIS_PRINTF("FindElementFromFace: Face buffer size: %d\n", FaceIdBuffer.size());
//     
////    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
////                 FaceIdBuffer[faceId].CommonElements[0].Id,
////                 FaceIdBuffer[faceId].CommonElements[0].Type,
////                 FaceIdBuffer[faceId].CommonElements[1].Id,
////                 FaceIdBuffer[faceId].CommonElements[1].Type);
//    // The test point is "inside" the element if d >= 0
//    ElVis::ElementId id;
//    if( d >= 0 )
//    {
//        
//        id = FaceIdBuffer[payload_v.FaceId].CommonElements[0];
//    }
//    else
//    {
//        id = FaceIdBuffer[payload_v.FaceId].CommonElements[1];
//    }
//    
//    findElementPayload.elementId = id.Id;
//    findElementPayload.elementType = id.Type;
//    //findElementPayload.IntersectionPoint = pointOnFace;
//    ELVIS_PRINTF("FindElementFromFace: Element Id %d and Type %d\n", id.Id, id.Type);
//    ELVIS_PRINTF("FindElementFromFace: Inside id %d and tpye %d, outside id %d and type %d\n", 
//        FaceIdBuffer[payload_v.FaceId].CommonElements[0].Id,
//        FaceIdBuffer[payload_v.FaceId].CommonElements[0].Type,
//        FaceIdBuffer[payload_v.FaceId].CommonElements[1].Id,
//        FaceIdBuffer[payload_v.FaceId].CommonElements[1].Type);
//    return findElementPayload;
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
