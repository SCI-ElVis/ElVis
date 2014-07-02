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

/// Storage for solution coefficients: loop over field, then expansion, then
/// coefficients themselves.
rtBuffer<ElVisFloat>  SolutionBuffer;

/// Coefficient offset buffer; identifies element IDs with offset in the
/// solution buffer. Currently assumes that all fields have the same number of
/// coefficients.
rtBuffer<int>         CoeffOffsetBuffer;

/// Number of modes for each element. 
rtBuffer<uint3>       ExpNumModesBuffer;

/// Linear coordinates of each element. Vertices are stored element by element.
rtBuffer<ElVisFloat3> CoordBuffer;

/// Coordinate offset buffer; identifies element IDs with storage positions in
/// the coordinate buffer.
rtBuffer<int>         CoordOffsetBuffer;

/// Contains face coefficients for any curved faces.
rtBuffer<ElVisFloat3> FaceCoeffsBuffer;

/// Contains offset of a curved face within the buffer.
rtBuffer<int>         FaceCoeffsOffsetBuffer;

// Record number of curved faces
rtDeclareVariable(int, nCurvedFaces, , );

#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/Hexahedron.cu>

#include <LibUtilities/BasicUtils/ShapeType.hpp>

// Everything
ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceOptiX(
    int                                elementId,
    int                                elementType,
    const WorldPoint&                  wp,
    ElVis::ReferencePointParameterType referenceType,
    ReferencePoint&                    result)
{
    if (referenceType != ElVis::eReferencePointIsValid)
    {
        if (elementType == Nektar::LibUtilities::eHexahedron)
        {
            result = TransformWorldToReferenceHex(elementId, wp);
        }
        else
        {
            return eInvalidElementType;
        }
    }

    return eNoError;
}

// Everything
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
    result = MAKE_FLOAT(0.0);

    if (elementType == Nektar::LibUtilities::eHexahedron)
    {
        result = EvaluateHexAtReferencePoint(elementId, tp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }

    return returnVal;
}

// Curved? may not be used
ELVIS_DEVICE ElVisError IsValidFaceCoordinate(
    GlobalFaceIdx             faceId,
    const FaceReferencePoint& point,
    bool&                     result)
{
    result = point.x >= MAKE_FLOAT(-1.0) &&
             point.x <= MAKE_FLOAT( 1.0) &&
             point.y >= MAKE_FLOAT(-1.0) &&
             point.y <= MAKE_FLOAT( 1.0);
    return eNoError;
}

// Curved
template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(
    GlobalFaceIdx faceId,
    const FaceReferencePoint& p,
    T& dx_dr, T& dx_ds,
    T& dy_dr, T& dy_ds,
    T& dz_dr, T& dz_ds)
{
    CurvedFaceIdx Idx = ConvertToCurvedFaceIdx(faceId);

    if (Idx.Value >= nCurvedFaces)
    {
        rtPrintf("############ EvaluateFace Idx.Value(%d) >= nCurvedFace(%d)", Idx.Value, nCurvedFaces);
        return eFieldNotDefinedOnFace;
    }

    return eNoError;
}

// Planar
ELVIS_DEVICE ElVisError GetFaceNormal(
    const WorldPoint& pointOnFace,
    GlobalFaceIdx     faceId,
    ElVisFloat3&      result)
{
    PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(faceId);
    if( planarFaceIdx.Value >= 0 )
    {
        result = MakeFloat3(PlanarFaceNormalBuffer[planarFaceIdx.Value]);
        return eNoError;
    }
    else
    {
        return eNoError;
    }
}

// Curved
ELVIS_DEVICE ElVisError GetFaceNormal(
    const WorldPoint&         pointOnFace,
    const FaceReferencePoint& refPoint,
    GlobalFaceIdx             faceId,
    ElVisFloat3&              result)
{
    result.x = MAKE_FLOAT(1.0);
    result.y = MAKE_FLOAT(0.0);
    result.z = MAKE_FLOAT(0.0);
    return eNoError;
}

// Returns the world space position (x,y,z) for face faceId and parametric
// coordinates (r,s).

// Curved
ELVIS_DEVICE ElVisError EvaluateFace(
    GlobalFaceIdx             faceId,
    const FaceReferencePoint& refPoint,
    WorldPoint&               result)
{
    return eNoError;
}

// Legacy
ELVIS_DEVICE ElVisError SampleReferenceGradientOptiX(
    int                   elementId,
    int                   elementType,
    int                   fieldId,
    const ReferencePoint& refPoint,
    ElVisFloat3&          gradient)
{
    gradient.x = MAKE_FLOAT(0.0);
    gradient.y = MAKE_FLOAT(0.0);
    gradient.z = MAKE_FLOAT(0.0);
    return eNoError;
}

// Legacy
ELVIS_DEVICE ElVisError SampleGeometryMappingJacobianOptiX(
    int                   elementId,
    int                   elementType,
    const ReferencePoint& refPoint,
    ElVisFloat*           J)
{
    return eNoError;
}

// Curved: reference/starting point for Newton algorithm
ELVIS_DEVICE ElVisError getStartingReferencePointForNewtonIteration(const CurvedFaceIdx& idx, ElVisFloat2& startingPoint)
{
    startingPoint.x = 0.0;
    startingPoint.y = 0.0;
    return eNoError;
}

// Curved: stop point from leaving the element
ELVIS_DEVICE ElVisError adjustNewtonStepToKeepReferencePointOnFace(const CurvedFaceIdx& idx, ElVisFloat3& newPoint)
{
    newPoint.x = max(min(newPoint.x, 1.0), -1.0);
    newPoint.y = max(min(newPoint.x, 1.0), -1.0);
    newPoint.z = max(min(newPoint.x, 1.0), -1.0);
    return eNoError;
}

#endif
