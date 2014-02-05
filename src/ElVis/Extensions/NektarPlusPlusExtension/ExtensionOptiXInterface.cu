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

#include <ElVis/Extensions/NektarPlusPlusExtension/typedefs.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/OptixHexahedron.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/OptixQuad.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/OptixTriangle.cu>

rtBuffer<ElVisFloat4> FaceNormalBuffer;

// Returns the world space position (x,y,z) for face faceId and parametric coordinates (r,s).
ELVIS_DEVICE ElVisError EvaluateFace(int faceId, const FaceReferencePoint& refPoint,
                               WorldPoint& result)
{
    ElVisFloat4 v0 = PlanarFaceVertexBuffer[4*faceId];
    ElVisFloat4 v1 = PlanarFaceVertexBuffer[4*faceId+1];
    ElVisFloat4 v2 = PlanarFaceVertexBuffer[4*faceId+2];
    ElVisFloat4 v3 = PlanarFaceVertexBuffer[4*faceId+3];

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
    if( elementType == Nektar::SpatialDomains::eHexahedron )
    {
        result = TransformWorldToTensor(elementId, wp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

ELVIS_DEVICE ElVisError SampleReferenceGradientOptiX(int elementId, int elementType, int fieldId, const ReferencePoint& refPoint, ElVisFloat3& gradient)
{
    ElVisError returnVal = eNoError;
    if( elementType == Nektar::SpatialDomains::eHexahedron )
    {
        gradient.x = EvaluateHexGradientDir1AtTensorPoint(elementId, refPoint.x, refPoint.y, refPoint.z);
        gradient.y = EvaluateHexGradientDir2AtTensorPoint(elementId, refPoint.x, refPoint.y, refPoint.z);
        gradient.z = EvaluateHexGradientDir3AtTensorPoint(elementId, refPoint.x, refPoint.y, refPoint.z);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

ELVIS_DEVICE ElVisError SampleGeometryMappingJacobianOptiX(int elementId, int elementType, const ReferencePoint& refPoint, ElVisFloat* J)
{
    ElVisError returnVal = eNoError;
    if( elementType ==  Nektar::SpatialDomains::eHexahedron )
    {
        calculateTensorToWorldSpaceMappingJacobian(elementId, refPoint, J);
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
    if( elementType == Nektar::SpatialDomains::eHexahedron )
    {
        result = EvaluateNektarPlusPlusHexAtTensorPoint(elementId, tp);
    }
    else
    {
        returnVal = eInvalidElementType;
    }
    return returnVal;
}

ELVIS_DEVICE ElVisError GetNumberOfVerticesForFace(int faceId, int& result)
{
    ElVisFloat4 v2 = PlanarFaceVertexBuffer[4*faceId+2];
    ElVisFloat4 v3 = PlanarFaceVertexBuffer[4*faceId+3];

    if( v2 == v3 )
    {
        result = 3;
    }
    else
    {
        result = 4;
    }

    return eNoError;
}

ELVIS_DEVICE ElVisError GetFaceVertex(int faceId, int vertexId, ElVisFloat4& result)
{
    result = PlanarFaceVertexBuffer[4*faceId+vertexId];
    return eNoError;
}

ELVIS_DEVICE ElVisError IsValidFaceCoordinate(int faceId, const FaceReferencePoint&, bool& result)
{
    //result = true;
    result = true;
    return eNoError;
}

template<typename T>
ELVIS_DEVICE ElVisError EvaluateFaceJacobian(int faceId, const FaceReferencePoint& p,
                                             T& dx_dr, T& dx_ds,
                                             T& dy_dr, T& dy_ds,
                                             T& dz_dr, T& dz_ds)
{
    ElVisFloat4 v0 = PlanarFaceVertexBuffer[4*faceId];
    ElVisFloat4 v1 = PlanarFaceVertexBuffer[4*faceId+1];
    ElVisFloat4 v2 = PlanarFaceVertexBuffer[4*faceId+2];
    ElVisFloat4 v3 = PlanarFaceVertexBuffer[4*faceId+3];

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


ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat3& pointOnFace, int faceId, ElVisFloat3& result)
{
    result = MakeFloat3(FaceNormalBuffer[faceId]);
    return eNoError;
}

ELVIS_DEVICE ElVisError GetFaceNormal(const ElVisFloat2& referencePointOnFace, const ElVisFloat3& worldPointOnFace, int faceId, ElVisFloat3& result)
{
    result = MakeFloat3(FaceNormalBuffer[faceId]);
    return eNoError;
}


#endif
