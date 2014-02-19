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

#ifndef ELVIS_EXTENSION_JACOBI_EXTENSION_OPTIX_INTERFACE_CU
#define ELVIS_EXTENSION_JACOBI_EXTENSION_OPTIX_INTERFACE_CU

#include <ElVis/Extensions/JacobiExtension/OptiXHexahedron.cu>
#include <ElVis/Extensions/JacobiExtension/OptiXPrism.cu>


// Returns the world space position (x,y,z) for face faceId and parametric coordinates (r,s).
ELVIS_DEVICE ElVisError EvaluateFace(GlobalFaceIdx globalFaceIdx, const FaceReferencePoint& refPoint,
                               WorldPoint& result)
{
    //ElVisFloat4 v0 = VertexBuffer[4*faceId];
    //ElVisFloat4 v1 = VertexBuffer[4*faceId+1];
    //ElVisFloat4 v2 = VertexBuffer[4*faceId+2];
    //ElVisFloat4 v3 = VertexBuffer[4*faceId+3];
    PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(globalFaceIdx);
    if( planarFaceIdx.Value < 0 ) return eInvalidFaceId;
    ElVisFloat4 v0, v1, v2, v3;
    GetPlanarFaceVertex(planarFaceIdx, 0, v0);
    GetPlanarFaceVertex(planarFaceIdx, 1, v1);
    GetPlanarFaceVertex(planarFaceIdx, 2, v2);
    GetPlanarFaceVertex(planarFaceIdx, 3, v3);

    ElVisFloat r = refPoint.x;
    ElVisFloat s = refPoint.y;

    if( v2 == v3 )
    {
        // Triangle.
        ElVisFloat s0 = ( MAKE_FLOAT(1.0)-r ) * (MAKE_FLOAT(1.0) - s) * MAKE_FLOAT(.25);
        ElVisFloat s1 = ( MAKE_FLOAT(1.0)+r ) * (MAKE_FLOAT(1.0) - s) * MAKE_FLOAT(.25);
        ElVisFloat s2 =  (MAKE_FLOAT(1.0) + s) * MAKE_FLOAT(.5);

        result.x = s0*v0.x + s1*v1.x + s2*v2.x;
        result.y = s0*v0.y + s1*v1.y + s2*v2.y;
        result.z = s0*v0.z + s1*v1.z + s2*v2.z;
    }
    else
    {
        // Quad
        ElVisFloat s0 = ( MAKE_FLOAT(1.0)-r ) * (MAKE_FLOAT(1.0) - s) * MAKE_FLOAT(.25);
        ElVisFloat s1 = ( MAKE_FLOAT(1.0)+r ) * (MAKE_FLOAT(1.0) - s) * MAKE_FLOAT(.25);
        ElVisFloat s3 = ( MAKE_FLOAT(1.0)-r ) * (MAKE_FLOAT(1.0) + s) * MAKE_FLOAT(.25);
        ElVisFloat s2 = ( MAKE_FLOAT(1.0)+r ) * (MAKE_FLOAT(1.0) + s) * MAKE_FLOAT(.25);
        result.x = s0*v0.x + s1*v1.x + s2*v2.x + s3*v3.x;
        result.y = s0*v0.y + s1*v1.y + s2*v2.y + s3*v3.y;
        result.z = s0*v0.z + s1*v1.z + s2*v2.z + s3*v3.z;
    }
    return eNoError;
}

ELVIS_DEVICE ElVisError ConvertWorldToReferenceSpaceOptiX(int elementId, int elementType, const WorldPoint& wp,
                                                          ElVis::ReferencePointParameterType referenceType, ReferencePoint& result)
{
    ElVisError returnVal = eNoError;
    if( elementType == 0 )
    {
        result = TransformWorldToTensor(&HexVertexBuffer[0], elementId, wp);
    }
    else if( elementType == 1 )
    {
        result = TransformPrismWorldToTensor(&PrismVertexBuffer[0],elementId, wp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

template<typename PointType, typename ResultType>
ELVIS_DEVICE ElVisError SampleScalarFieldAtReferencePointOptiX(int elementId, int elementType, int fieldId,
                                                               const PointType& worldPoint,
                                                               const PointType& tp,
                                                               ResultType& result)
{
    ElVisError returnVal = eNoError;
    if( elementType == 0 )
    {
        uint3 degree = HexDegrees[elementId];

        uint coefficientIndex = HexCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(HexCoefficients[coefficientIndex]);

        result = EvaluateHexFieldAtTensorPoint<ElVisFloat>(degree, tp.x, tp.y, tp.z, coeffs);
    }
    else if( elementType == 1 )
    {
        uint3 degree = PrismDegrees[elementId];

        uint coefficientIndex = PrismCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);

        result = EvaluatePrismFieldAtTensorPoint(degree, tp.x, tp.y, tp.z, coeffs);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}



ELVIS_DEVICE ElVisError IsValidFaceCoordinate(GlobalFaceIdx faceId, const FaceReferencePoint&, bool& result)
{
    result = true;
    return eNoError;
}

template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(GlobalFaceIdx globalFaceIdx, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{
    //ElVisFloat4 v0 = VertexBuffer[4*faceId];
    //ElVisFloat4 v1 = VertexBuffer[4*faceId+1];
    //ElVisFloat4 v2 = VertexBuffer[4*faceId+2];
    //ElVisFloat4 v3 = VertexBuffer[4*faceId+3];
    PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(globalFaceIdx);
    if( planarFaceIdx.Value < 0 ) return eInvalidFaceId;

    ElVisFloat4 v0, v1, v2, v3;
    GetPlanarFaceVertex(planarFaceIdx, 0, v0);
    GetPlanarFaceVertex(planarFaceIdx, 1, v1);
    GetPlanarFaceVertex(planarFaceIdx, 2, v2);
    GetPlanarFaceVertex(planarFaceIdx, 3, v3);

    const T& r = p.x;
    const T& s = p.y;

    if( v2 == v3 )
    {
        // df/dr
        {
            T s0 = -(MAKE_FLOAT(1.0)-s)*MAKE_FLOAT(.25);
            T s1 = (MAKE_FLOAT(1.0)-s)*MAKE_FLOAT(.25);
            T s2 = MAKE_FLOAT(0.0);

            dx_dr = v0.x*s0 + v1.x*s1 + v2.x*s2;
            dy_dr = v0.y*s0 + v1.y*s1 + v2.y*s2;
            dz_dr = v0.z*s0 + v1.z*s1 + v2.z*s2;
        }

        //df/ds
        {
            T s0 = -(MAKE_FLOAT(1.0)-r)*MAKE_FLOAT(.25);
            T s1 = -(MAKE_FLOAT(1.0)+r)*MAKE_FLOAT(.25);
            T s2 = MAKE_FLOAT(.5);

            dx_ds = v0.x*s0 + v1.x*s1 + v2.x*s2;
            dy_ds = v0.y*s0 + v1.y*s1 + v2.y*s2;
            dz_ds = v0.z*s0 + v1.z*s1 + v2.z*s2;
        }
    }
    else
    {
        // df/dr
        {
            T s0 = -(MAKE_FLOAT(1.0)-s)*MAKE_FLOAT(.25);
            T s1 = (MAKE_FLOAT(1.0)-s)*MAKE_FLOAT(.25);
            T s2 = (MAKE_FLOAT(1.0)+s)*MAKE_FLOAT(.25);
            T s3 = -(MAKE_FLOAT(1.0)+s)*MAKE_FLOAT(.25);

            dx_dr = v0.x*s0 + v1.x*s1 + v2.x*s2 + v3.x*s3;
            dy_dr = v0.y*s0 + v1.y*s1 + v2.y*s2 + v3.y*s3;
            dz_dr = v0.z*s0 + v1.z*s1 + v2.z*s2 + v3.z*s3;
        }

        // df/ds
        {
            T s0 = -(MAKE_FLOAT(1.0) - r) * MAKE_FLOAT(.25);
            T s1 = -(MAKE_FLOAT(1.0) + r) * MAKE_FLOAT(.25);
            T s2 = (MAKE_FLOAT(1.0) + r) * MAKE_FLOAT(.25);
            T s3 = (MAKE_FLOAT(1.0) - r) * MAKE_FLOAT(.25);

            dx_ds = v0.x*s0 + v1.x*s1 + v2.x*s2 + v3.x*s3;
            dy_ds = v0.y*s0 + v1.y*s1 + v2.y*s2 + v3.y*s3;
            dz_ds = v0.z*s0 + v1.z*s1 + v2.z*s2 + v3.z*s3;
        }
    }
    return eNoError;
}


ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& result)
{
    PlanarFaceIdx planarFaceIdx = ConvertToPlanarFaceIdx(globalFaceIdx);
    if( planarFaceIdx.Value > 0 )
    {
      result = MakeFloat3(PlanarFaceNormalBuffer[planarFaceIdx.Value]);
      return eNoError;
    }
    else
    {
      return eInvalidFaceId;
    }
}

ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat2& referencePointOnFace, const ElVisFloat3& worldPointOnFace, GlobalFaceIdx globalFaceIdx, ElVisFloat3& result)
{
    result.x = MAKE_FLOAT(0.0);
    result.y = MAKE_FLOAT(0.0);
    result.z = MAKE_FLOAT(0.0);

    return eNoError;
}

ELVIS_DEVICE ElVisFloat3 CalculateTensorGradient(unsigned int elementId, unsigned int elementType, int fieldId, const TensorPoint& p)
{
    ElVisFloat3 result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    if( elementType == 0 )
    {
        uint3 degree = HexDegrees[elementId];

        uint coefficientIndex = HexCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(HexCoefficients[coefficientIndex]);

        result.x = EvaluateHexGradientDir1AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
        result.y = EvaluateHexGradientDir2AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
        result.z = EvaluateHexGradientDir3AtTensorPoint(degree, p.x, p.y, p.z, coeffs);
    }
    else if( elementType == 1 )
    {
        uint3 degree = PrismDegrees[elementId];

        uint coefficientIndex = PrismCoefficientIndices[elementId];
        ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);

        result.x = EvaluatePrismGradientDir1AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
        result.y = EvaluatePrismGradientDir2AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
        result.z = EvaluatePrismGradientDir3AtTensorPoint<ElVisFloat>(degree, p.x, p.y, p.z, coeffs);
    }
    return result;
}

ELVIS_DEVICE ElVisError SampleReferenceGradientOptiX(int elementId, int elementType, int fieldId, const ReferencePoint& refPoint, ElVisFloat3& gradient)
{
    gradient = CalculateTensorGradient(elementId, elementType, fieldId, refPoint);
    return eNoError;
}

ELVIS_DEVICE ElVisError SampleGeometryMappingJacobianOptiX(int elementId, int elementType, const ReferencePoint& refPoint, ElVisFloat* J)
{
    if( elementType == 0 )
    {
        calculateTensorToWorldSpaceMappingJacobian(&HexVertexBuffer[0], elementId, refPoint, J);
    }
    else
    {
        GetPrismWorldToReferenceJacobian(&PrismVertexBuffer[0], elementId, refPoint, J);
    }
    return eNoError;
}

#endif
