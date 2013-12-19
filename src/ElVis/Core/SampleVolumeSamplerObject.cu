
#ifndef SAMPLE_VOLUME_SAMPLER_OBJECT_CU
#define SAMPLE_VOLUME_SAMPLER_OBJECT_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>
#include "CutSurfacePayloads.cu"
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/FindElement.cu>

/// \brief If a cut surface is located outside the volume, then we do not want it to be considered for the
///        closest hit.  We only care about surfaces inside the volume.
RT_PROGRAM void IgnoreCutSurfacesOutOfVolume()
{
    const ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + closest_t * MakeFloat3(ray.direction);

    ElVisFloat3 minBox = VolumeMinExtent - MAKE_FLOAT(.1);
    ElVisFloat3 maxBox = VolumeMaxExtent + MAKE_FLOAT(.1);

    if( !PointInBox(intersectionPoint, minBox, maxBox) )
    {
        ELVIS_PRINTF("IgnoreCutSurfacesOutOfVolume: Ignoring intersection at %f\n", closest_t);
        rtIgnoreIntersection();
    }
}

/// \brief When a cut-surface is encountered, this closest hit program is executed to determine the
///        value of the currently selected field at that point.
RT_PROGRAM void SamplerVolumeClosestHit()
{
    ELVIS_PRINTF("SamplerVolumeClosestHit: Evaluating surface at t=%f\n", closest_t);
    const ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + closest_t * MakeFloat3(ray.direction);   
    //ElementFinderPayload findElementPayload = FindElement(intersectionPoint);
    ElementFinderPayload findElementPayload = FindElementFromFace(intersectionPoint);
    
    // Now use the extension interface to sample the field.
    if( findElementPayload.elementId >= 0 )
    {
        ELVIS_PRINTF("SamplerVolumeClosestHit: Element id is %d\n", findElementPayload.elementId);
        payload.scalarValue = EvaluateFieldOptiX(findElementPayload.elementId, findElementPayload.elementType, FieldId, intersectionPoint, findElementPayload.ReferencePointType, findElementPayload.ReferenceIntersectionPoint);
        payload.isValid = true;

        payload.Normal = normal;
        payload.IntersectionPoint = intersectionPoint;
        payload.IntersectionT = closest_t;
        payload.elementId = findElementPayload.elementId;
        payload.elementType = findElementPayload.elementType;
        //ELVIS_PRINTF("Payload is valid %d.\n", (payload.isValid ? 1 : 0));
    }
    else
    {
        ELVIS_PRINTF("SamplerVolumeClosestHit: No element found.\n");
        payload.scalarValue = ELVIS_FLOAT_MAX;
        payload.isValid = false;

        payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        payload.IntersectionPoint = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        payload.IntersectionT = MAKE_FLOAT(-1.0);
        payload.elementId = -1;
        payload.elementType = -1;
    }
}

rtDeclareVariable(int, faceElementId, , );
rtDeclareVariable(int, faceElementType, , );

//RT_PROGRAM void SamplerFaceClosestHit()
//{
//    ELVIS_PRINTF("SamplerFaceClosestHit: face id is %d\n", intersectedFaceGlobalIdx);
//    const ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + closest_t * MakeFloat3(ray.direction);
//    ElementFinderPayload findElementPayload = FindElement(intersectionPoint);


//    // Now use the extension interface to sample the field.
//    if( findElementPayload.elementId != -1 )
//    {
//        ELVIS_PRINTF("SamplerFaceClosestHit: Element id is %d and face id is %d\n", findElementPayload.elementId, intersectedFaceGlobalIdx);
//        payload.scalarValue = EvaluateFieldOptiX(findElementPayload.elementId, findElementPayload.elementType, FieldId, intersectionPoint, ElVis::eReferencePointIsValid, findElementPayload.ReferenceIntersectionPoint);
//        payload.isValid = true;

////        payload.Normal = normal;
//        payload.IntersectionPoint = intersectionPoint;
//        payload.IntersectionT = closest_t;
//        payload.elementId = findElementPayload.elementId;
//        payload.elementType = findElementPayload.elementType;
//        ELVIS_PRINTF("Payload is valid %d.\n", (payload.isValid ? 1 : 0));
//    }
//    else
//    {
//        ELVIS_PRINTF("SamplerFaceClosestHit: No element found.\n");
//        payload.scalarValue = ELVIS_FLOAT_MAX;
//        payload.isValid = false;

//        payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
//        payload.IntersectionPoint = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
//        payload.IntersectionT = MAKE_FLOAT(-1.0);
//        payload.elementId = 0;
//        payload.elementType = 0;
//    }
//}


RT_PROGRAM void SamplerFaceClosestHit()
{
    // The geometry instance associated with the intersection corresponds
    // has the faceElementId and faceElementType above to evaluate the field.
    ELVIS_PRINTF("SamplerFaceClosestHit\n");
//    ELVIS_PRINTF("SamplerFaceClosestHit Origin (%2.15f, %2.15f, %2.15f), closest t %2.15f, direction (%2.15f, %2.15f, %2.15f)\n",
//                 ray.origin.x, ray.origin.y, ray.origin.z,
//                 closest_t,
//                 ray.direction.x, ray.direction.y, ray.direction.z);
    const ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + closest_t * MakeFloat3(ray.direction);
//    ELVIS_PRINTF("SamplerFaceClosestHit Intersection point (%2.15f, %2.15f, %2.15f)\n",
//                 intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);

    ElVis::ElementId id;

    ElVisFloat3 testPoint = MakeFloat3(ray.origin);
    ElVisFloat3 n;
    if( faceIntersectionReferencePointIsValid )
    {
        // We do know the face reference coordinates.
        ELVIS_PRINTF("Have reference point.\n");
        id = FindElement(testPoint, intersectionPoint, faceIntersectionReferencePoint, intersectedFaceGlobalIdx, n);
    }
    else
    {
        // We don't know the reference coordinate (because the intersection program didn't
        // provide them).
        ELVIS_PRINTF("Don't have reference point.\n");
        id = FindElement(testPoint, intersectionPoint, intersectedFaceGlobalIdx, n);
    }


    // Find the element associated with this face.
    ELVIS_PRINTF("SamplerFaceClosestHit: Element Id %d, Element Type %d, FieldID %d, Normal (%f, %f, %f)\n", id.Id, id.Type, FieldId, n.x, n.y, n.z);
    payload.scalarValue = EvaluateFieldOptiX(id.Id, id.Type, FieldId, intersectionPoint);
    ELVIS_PRINTF("SamplerFaceClosestHit: Scalar Value %2.15f\n", payload.scalarValue);
    payload.isValid = true;

    payload.Normal = n;
    payload.IntersectionPoint = intersectionPoint;
    payload.IntersectionT = closest_t;
    payload.elementId = id.Id;
    payload.elementType = id.Type;
}



rtDeclareVariable(ElVisFloat3, SampleOntoNrrdH, ,);
rtBuffer<ElVisFloat, 2> SampledOntoNrrdSamples;
rtDeclareVariable(ElVisFloat3, SampleOntoNrrdMinExtent, ,);
rtDeclareVariable(int, SampleOntoNrrdPlane, , );
rtDeclareVariable(ElVisFloat, SampleOntoNrrdMissValue, , );
RT_PROGRAM void SampleOntoNrrd()
{
    ElVisFloat3 intersectionPoint = MakeFloat3(
                SampleOntoNrrdMinExtent.x + SampleOntoNrrdH.x*launch_index.x,
                SampleOntoNrrdMinExtent.y + SampleOntoNrrdH.y*launch_index.y,
                SampleOntoNrrdMinExtent.z + SampleOntoNrrdH.z*SampleOntoNrrdPlane);
    ElementFinderPayload findElementPayload = FindElement(intersectionPoint);

    // Now use the extension interface to sample the field.
    if( findElementPayload.elementId != -1 )
    {
        SampledOntoNrrdSamples[launch_index] = EvaluateFieldOptiX(findElementPayload.elementId, findElementPayload.elementType, FieldId, intersectionPoint, findElementPayload.ReferencePointType, findElementPayload.ReferenceIntersectionPoint);
    }
    else
    {
        SampledOntoNrrdSamples[launch_index] = SampleOntoNrrdMissValue;
    }
}

#endif
