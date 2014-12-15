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
#include <ElVis/Core/ReferencePointParameter.h>

/// \brief Finds the element that encloses a point based on one of the element's faces.
/// \param[in] p The point that the element encloses.
/// \param[in] direction The direction of a ray cast from p to find a face.
/// \param[in] payload_v The payload that resulted from casting a ray to find a face.
/// \param[out] rFindElementPayload Populated with the type and element id of the enclosing element.
/// \returns true if the enclosing element was found, false otherwise.
///
/// Due to numerical issues, it is possible for this method to return false even if the point is enclosed by an
/// element.
__device__ __forceinline__  bool findElementFromFace(const ElVisFloat3& p, const ElVisFloat3& direction, const VolumeRenderingPayload& payload_v,
                                                ElementFinderPayload& rFindElementPayload)
{
    rFindElementPayload.Initialize(p);
    if( !payload_v.FoundIntersection )
    {
        ELVIS_PRINTF("FindElementFromFace: Did not find element intersection.\n");
        return false;
    }
    else
    {
        ELVIS_PRINTF("FindElementFromFace: Found element intersection.\n");
    }

    ElVisFloat3 faceNormal;
    ElVisFloat3 pointOnFace = p + payload_v.IntersectionT*direction;

    if( payload_v.FaceReferecePointIsValid )
    {
        ELVIS_PRINTF("FindElementFromFace: Calling reference point aware face normal (%f, %f)\n",
                     payload_v.FaceReferencePoint.x, payload_v.FaceReferencePoint.y);
        GetFaceNormal(pointOnFace, payload_v.FaceReferencePoint, payload_v.FaceId, faceNormal);
    }
    else
    {
        GetFaceNormal(pointOnFace, payload_v.FaceId, faceNormal);
    }

    ElVisFloat3 vectorToPointOnFace = normalize( pointOnFace - p ) ;//p - pointOnFace;

    //Normalize the vectors before computing the dot product
    faceNormal = normalize(faceNormal);

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

    ELVIS_PRINTF("FindElementFromFace: Face Id %d, dot %2.15f (positive inside)\n", payload_v.FaceId.Value, d);
    ELVIS_PRINTF("FindElementFromFace: Inside id %d and tpye %d, outside id %d and type %d\n",
                 GetFaceInfo(payload_v.FaceId).CommonElements[0].Id,
                 GetFaceInfo(payload_v.FaceId).CommonElements[0].Type,
                 GetFaceInfo(payload_v.FaceId).CommonElements[1].Id,
                 GetFaceInfo(payload_v.FaceId).CommonElements[1].Type);
    ELVIS_PRINTF("FindElementFromFace: Ray direction (%2.15f, %2.15f, %2.15f)\n",direction.x, direction.y, direction.z);
    ELVIS_PRINTF("FindElementFromFace: Normal (%2.15f, %2.15f, %2.15f), point on face (%2.15f, %2.15f, %2.15f)\n",
                 faceNormal.x, faceNormal.y, faceNormal.z,
                 pointOnFace.x, pointOnFace.y, pointOnFace.z);
    ELVIS_PRINTF("FindElementFromFace: Vector (%2.15f, %2.15f, %2.15f), point in elem (%2.15f, %2.15f, %2.15f)\n",
                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z,
                 p.x, p.y, p.z);

    //ELVIS_PRINTF("FindElementFromFace: Face buffer size: %d\n", FaceInfoBuffer.size());
     
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceInfoBuffer[faceId].CommonElements[0].Id,
//                 FaceInfoBuffer[faceId].CommonElements[0].Type,
//                 FaceInfoBuffer[faceId].CommonElements[1].Id,
//                 FaceInfoBuffer[faceId].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    ElVis::ElementId ids[2];
    if( d >= 0 )
    {   
        ids[0] = GetFaceInfo(payload_v.FaceId).CommonElements[0];
        ids[1] = GetFaceInfo(payload_v.FaceId).CommonElements[1];
    }
    else
    {
        ids[0] = GetFaceInfo(payload_v.FaceId).CommonElements[1];
        ids[1] = GetFaceInfo(payload_v.FaceId).CommonElements[0];
    }
    
    for(int i = 0; i < 2; ++i)
    {
        ReferencePoint refPoint;
        ElVisError invertResult = ConvertWorldToReferenceSpaceOptiX(ids[i].Id, ids[i].Type,
                                                                    p, ElVis::eReferencePointIsInvalid, refPoint);
        int id = ids[i].Id;
        int type = ids[i].Type;

        if( invertResult == eNoError )
        {
            ELVIS_PRINTF("FindElementFromFace: Valid reference inversion for Element %d and Type %d with inversion (%f, %f, %f)\n",
                         id, type, refPoint.x, refPoint.y, refPoint.z);
            rFindElementPayload.elementId = ids[i].Id;
            rFindElementPayload.elementType = ids[i].Type;
            return true;
        }
        else
        {
            //ELVIS_PRINTF("FindElementFromFace: Invalid reference inversion for Element %d and Type %d\n", id, type);
        }
    }

    return false;
}

/// \brief Finds the element enclosing point p
__device__ __forceinline__ ElementFinderPayload FindElementFromFace(const ElVisFloat3& p)
{
    // Random direction since rays in any direction should intersect the element.  We did have some accuracy
    // issues with a direciton of (1, 0, 0) with axis-aligned elements.  This direction won't solve that
    // problem, but it does make it less likely to occur.
    ElVisFloat3 direction = MakeFloat3(ray.direction);

    //Assume that a 2D model is only on a z constant plane.
    if( ModelDimension == 2 ) direction.z = 0;

    ELVIS_PRINTF("FindElementFromFace: direction (%f, %f, %f)\n", direction.x, direction.y, direction.z);
    ELVIS_PRINTF("FindElementFromFace: Looking for element that encloses point (%f, %f, %f)\n", p.x, p.y, p.z);

    VolumeRenderingPayload payload_v = FindNextFaceIntersection(p, direction);
   
    ELVIS_PRINTF("FindElementFromFace: First one Found=%d T=%f id=%d\n", payload_v.FoundIntersection,
        payload_v.IntersectionT, payload_v.FaceId.Value);

    ElementFinderPayload findElementPayload;
    bool foundElement = findElementFromFace(p, direction, payload_v, findElementPayload);
    if( payload_v.FoundIntersection && !foundElement )
    {
        payload_v.Initialize();
        direction = MakeFloat3(-direction.x, -direction.y, -direction.z);
        payload_v = FindNextFaceIntersection(p, direction);
        ELVIS_PRINTF("FindElementFromFace Try 2: Found=%d T=%f id=%d\n", payload_v.FoundIntersection,
            payload_v.IntersectionT, payload_v.FaceId.Value);
        foundElement = findElementFromFace(p, direction, payload_v, findElementPayload);
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


    //return findElementPayload;


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
//    ELVIS_PRINTF("FindElementFromFace: Face buffer size: %d\n", FaceInfoBuffer.size());
//     
////    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
////                 FaceInfoBuffer[faceId].CommonElements[0].Id,
////                 FaceInfoBuffer[faceId].CommonElements[0].Type,
////                 FaceInfoBuffer[faceId].CommonElements[1].Id,
////                 FaceInfoBuffer[faceId].CommonElements[1].Type);
//    // The test point is "inside" the element if d >= 0
//    ElVis::ElementId id;
//    if( d >= 0 )
//    {
//        
//        id = FaceInfoBuffer[payload_v.FaceId].CommonElements[0];
//    }
//    else
//    {
//        id = FaceInfoBuffer[payload_v.FaceId].CommonElements[1];
//    }
//    
//    findElementPayload.elementId = id.Id;
//    findElementPayload.elementType = id.Type;
//    //findElementPayload.IntersectionPoint = pointOnFace;
//    ELVIS_PRINTF("FindElementFromFace: Element Id %d and Type %d\n", id.Id, id.Type);
//    ELVIS_PRINTF("FindElementFromFace: Inside id %d and tpye %d, outside id %d and type %d\n", 
//        FaceInfoBuffer[payload_v.FaceId].CommonElements[0].Id,
//        FaceInfoBuffer[payload_v.FaceId].CommonElements[0].Type,
//        FaceInfoBuffer[payload_v.FaceId].CommonElements[1].Id,
//        FaceInfoBuffer[payload_v.FaceId].CommonElements[1].Type);
//    return findElementPayload;
}

// In this version, we don't know the reference space coordinates for the face intersection, so
// we rely on the simulation package to do the calculation for us.
__device__ ElVis::ElementId FindElement(const ElVisFloat3& testPoint, const ElVisFloat3& pointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& faceNormal)
{
    GetFaceNormal(pointOnFace, globalFaceIdx, faceNormal);
    ElVisFloat3 vectorToPointOnFace = testPoint - pointOnFace;

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

//    ELVIS_PRINTF("FindElement: Face Id %d, Normal (%f, %f, %f), test point (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", globalFaceIdx,
//                 faceNormal.x, faceNormal.y, faceNormal.z,
//                 testPoint.x, testPoint.y, testPoint.z,
//                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[0].Id,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[0].Type,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[1].Id,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    if( d >= 0 )
    {
        return GetFaceInfo(globalFaceIdx).CommonElements[0];
    }
    else
    {
        return GetFaceInfo(globalFaceIdx).CommonElements[1];
    }
}

__device__ ElVis::ElementId FindElement(const ElVisFloat3& testPoint, const ElVisFloat3& pointOnFace, const ElVisFloat2& referencePointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& faceNormal)
{
    GetFaceNormal(pointOnFace, referencePointOnFace, globalFaceIdx, faceNormal);
    ElVisFloat3 vectorToPointOnFace = testPoint - pointOnFace;

    ElVisFloat d = dot(faceNormal, vectorToPointOnFace);

//    ELVIS_PRINTF("FindElement: Face Id %d, Normal (%f, %f, %f), test point (%f, %f, %f) Vector (%f, %f, %f) dot %f (positive inside)\n", globalFaceIdx,
//                 faceNormal.x, faceNormal.y, faceNormal.z,
//                 testPoint.x, testPoint.y, testPoint.z,
//                 vectorToPointOnFace.x, vectorToPointOnFace.y, vectorToPointOnFace.z, d);
//    ELVIS_PRINTF("FindElement: Inside Element %d and type %d and outside element %d and type %d\n",
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[0].Id,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[0].Type,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[1].Id,
//                 FaceInfoBuffer[globalFaceIdx].CommonElements[1].Type);
    // The test point is "inside" the element if d >= 0
    if( d >= 0 )
    {
        return GetFaceInfo(globalFaceIdx).CommonElements[0];
    }
    else
    {
        return GetFaceInfo(globalFaceIdx).CommonElements[1];
    }
}

/// \brief Finds the element enclosing point p
__device__ __forceinline__ ElementFinderPayload FindElement(const ElVisFloat3& p)
{
  return FindElementFromFace(p);
}

#endif
