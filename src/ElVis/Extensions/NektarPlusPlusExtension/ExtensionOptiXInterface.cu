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

#ifndef ELVIS_NEKTAR_PLUS_PLUS_EXTENSION_EXTENSION_OPTIX_INTERFACE_CU
#define ELVIS_NEKTAR_PLUS_PLUS_EXTENSION_EXTENSION_OPTIX_INTERFACE_CU

#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/Float.cu>

rtBuffer<ElVisFloat4> FaceNormalBuffer;

ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceOptiX(
    int                                elementId,
    int                                elementType,
    const WorldPoint&                  wp,
    ElVis::ReferencePointParameterType referenceType,
    ReferencePoint&                    result)
{
    return eNoError;
}

template<typename PointType, typename ResultType>
ELVIS_DEVICE ElVisError SampleScalarFieldAtReferencePointOptiX(
    int              elementId,
    int              elementType,
    int              fieldId,
    const PointType& worldPoint,
    const PointType& tp,
    ResultType&      result)
{
    ElVisError returnVal = eNoError;
    /*
    if (elementType == Nektar::SpatialDomains::eHexahedron)
    {
        result = EvaluateNektarPlusPlusHexAtTensorPoint(elementId, tp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    */
    result = MAKE_FLOAT(0.0);
    return returnVal;
}

ELVIS_DEVICE ElVisError IsValidFaceCoordinate(
    GlobalFaceIdx             faceId,
    const FaceReferencePoint& point,
    bool&                     result)
{
    result = point.x >= MAKE_FLOAT(-1.0) &&
             point.x <= MAKE_FLOAT(1.0) &&
             point.y >= MAKE_FLOAT(-1.0) &&
             point.y <= MAKE_FLOAT(1.0);
    return eNoError;
}

template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(int faceId, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{
    return eNoError;
}

ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, int faceId, ElVisFloat3& result)
{
    PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(faceId);
    result = MakeFloat3(PlanarFaceNormalBuffer[planarFaceIdx.Value]);
    return eNoError;
}

ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat2& referencePointOnFace, const ElVisFloat3& worldPointOnFace, int faceId, ElVisFloat3& result)
{
    result.x = MAKE_FLOAT(1.0);
    result.y = MAKE_FLOAT(0.0);
    result.z = MAKE_FLOAT(0.0);
    return eNoError;
}

// Returns the world space position (x,y,z) for face faceId and parametric coordinates (r,s).
ELVIS_DEVICE ElVisError EvaluateFace(
    int                       faceId,
    const FaceReferencePoint& refPoint,
    WorldPoint&               result)
{
    return eNoError;
}

ELVIS_DEVICE ElVisError SampleReferenceGradientOptiX(
    int                   elementId,
    int                   elementType,
    int                   fieldId,
    const ReferencePoint& refPoint,
    ElVisFloat3&          gradient)
{
    ElVisError returnVal = eNoError;
    /*
    if (elementType == Nektar::SpatialDomains::eHexahedron)
    {
        gradient.x = EvaluateHexGradientDir1AtTensorPoint(
            elementId, refPoint.x, refPoint.y, refPoint.z);
        gradient.y = EvaluateHexGradientDir2AtTensorPoint(
            elementId, refPoint.x, refPoint.y, refPoint.z);
        gradient.z = EvaluateHexGradientDir3AtTensorPoint(
            elementId, refPoint.x, refPoint.y, refPoint.z);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    */
    gradient.x = MAKE_FLOAT(0.0);
    gradient.y = MAKE_FLOAT(0.0);
    gradient.z = MAKE_FLOAT(0.0);
    return returnVal;
}

ELVIS_DEVICE ElVisError SampleGeometryMappingJacobianOptiX(
    int                   elementId,
    int                   elementType,
    const ReferencePoint& refPoint,
    ElVisFloat*           J)
{
    ElVisError returnVal = eNoError;
    /*
    if( elementType ==  Nektar::SpatialDomains::eHexahedron )
    {
        calculateTensorToWorldSpaceMappingJacobian(elementId, refPoint, J);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    */
    return returnVal;
}

ELVIS_DEVICE ElVisError getStartingReferencePointForNewtonIteration(const CurvedFaceIdx& idx, ElVisFloat2& startingPoint)
{
    return eNoError;
}

ELVIS_DEVICE ElVisError adjustNewtonStepToKeepReferencePointOnFace(const CurvedFaceIdx& idx, ElVisFloat3& newPoint)
{
    return eNoError;
}

#endif
