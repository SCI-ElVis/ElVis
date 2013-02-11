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

#ifndef ELVIS_EXTENSIONS_JACOB_EXTENSION_OPTIX_PRISM_CU
#define ELVIS_EXTENSIONS_JACOB_EXTENSION_OPTIX_PRISM_CU

#include <ElVis/Core/matrix.cu>
#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_aabb.h>
#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/jacobi.cu>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Extensions/JacobiExtension/PrismCommon.cu>

// The vertices associated with this prism.
// Prism has 6 vertices.
rtBuffer<ElVisFloat4> PrismVertexBuffer;

// The vertices associated with each face.
// Faces 0-2 are quads and all four elements are used.
// Faces 3 and 4 are triangles
rtBuffer<uint4> Prismvertex_face_index;

// The planes associated with each face.
rtBuffer<ElVisFloat4> PrismPlaneBuffer;

// The coefficients to evaluate the scalar field.
rtBuffer<ElVisFloat> PrismCoefficients;
rtBuffer<uint> PrismCoefficientIndices;

rtBuffer<uint3> PrismDegrees;

rtDeclareVariable(int, intersectedPrismId, attribute IntersectedHex, );









__device__ __forceinline__ bool IntersectsPrismFace(int prismId, unsigned int faceNumber,
                               ElVisFloat4* p, const ElVisFloat3& origin, const ElVisFloat3& direction,
                               ElVisFloat& t)
{
    uint4 index = Prismvertex_face_index[faceNumber];
    bool result = false;
    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
    {
        result = FindPlaneIntersection(origin, direction, GetPrismPlane(&PrismPlaneBuffer[0], prismId, faceNumber), t);
    }

    return result;
}


// Intersection through half plane tests.
//// Determines if the given ray intersects the given hex.  Returns true if it does, false otherwise.  If an intersection is
//// found, t is the value of the closest intersection.
//__device__ bool PrismIntersection(const ElVisFloat3& origin, const ElVisFloat3& direction, int prismId, const ElVisFloat& closestT, ElVisFloat& t)
//{
//    t = closestT;
//    for(int faceId = 0; faceId < 5; ++faceId)
//    {
//        // Check to see if we intersect this face.
//        ElVisFloat plane_t;
//        bool intersectsFace = FindPlaneIntersection(origin, direction, GetPrismPlane(PrismPlaneBuffer, prismId, faceId), plane_t);

//        bool testInside = intersectsFace;
//        testInside &= (plane_t < t );
//        if( testInside )
//        {
//            WorldPoint intersectionPoint = origin + plane_t*direction;

//            bool insideOtherFaces = true;
//            for(int insideFaceId = 0; insideFaceId < 5; ++insideFaceId)
//            {
//                if( insideFaceId != faceId )
//                {
//                    ElVisFloat planeVal = EvaluatePlane(GetPrismPlane(PrismPlaneBuffer, prismId, insideFaceId), intersectionPoint);
//                    insideOtherFaces &= planeVal <= MAKE_FLOAT(0.0);
//                    if( !insideOtherFaces ) break;
//                }
//            }

//            if( insideOtherFaces )
//            {
//                t = plane_t;
//            }
//        }
//    }
//    return t != ELVIS_FLOAT_MAX;
//}

// Intersection through projection.
__device__ __forceinline__ void FindRayPrismIntersection(int prismId)
{
//    ElVisFloat3 origin = MakeFloat3(ray.origin);
//    ElVisFloat3 W = MakeFloat3(ray.direction);
//    normalize(W);
//    ElVisFloat3 U,V;
//    GenerateUVWCoordinateSystem(W, U, V);
//    // Project each vertex onto the ray's coorindate system
//    ElVis::Matrix<4,4> M1;
//    M1.getData()[0] = U.x;
//    M1.getData()[1] = U.y;
//    M1.getData()[2] = U.z;
//    M1.getData()[3] = MAKE_FLOAT(0.0);

//    M1.getData()[4] = V.x;
//    M1.getData()[5] = V.y;
//    M1.getData()[6] = V.z;
//    M1.getData()[7] = MAKE_FLOAT(0.0);

//    M1.getData()[8] = W.x;
//    M1.getData()[9] = W.y;
//    M1.getData()[10] = W.z;
//    M1.getData()[11] = MAKE_FLOAT(0.0);

//    M1.getData()[12] = MAKE_FLOAT(0.0);
//    M1.getData()[13] = MAKE_FLOAT(0.0);
//    M1.getData()[14] = MAKE_FLOAT(0.0);
//    M1.getData()[15] = MAKE_FLOAT(1.0);

//    ElVis::Matrix<4,4> M2;
//    M2.getData()[0] = MAKE_FLOAT(1.0);
//    M2.getData()[1] = MAKE_FLOAT(0.0);
//    M2.getData()[2] = MAKE_FLOAT(0.0);
//    M2.getData()[3] = -origin.x;

//    M2.getData()[4] = MAKE_FLOAT(0.0);
//    M2.getData()[5] = MAKE_FLOAT(1.0);
//    M2.getData()[6] = MAKE_FLOAT(0.0);
//    M2.getData()[7] = -origin.y;

//    M2.getData()[8] = MAKE_FLOAT(0.0);
//    M2.getData()[9] = MAKE_FLOAT(0.0);
//    M2.getData()[10] = MAKE_FLOAT(1.0);
//    M2.getData()[11] = -origin.z;

//    M2.getData()[12] = MAKE_FLOAT(0.0);
//    M2.getData()[13] = MAKE_FLOAT(0.0);
//    M2.getData()[14] = MAKE_FLOAT(0.0);
//    M2.getData()[15] = MAKE_FLOAT(1.0);

//    ElVis::Matrix<4,4> M3 = M1*M2;
//    // Fill in as p16 of pete's book.

//    ElVisFloat4 p[] = {
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 0),
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 1),
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 2),
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 3),
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 4),
//        M3*GetPrismVertex(&PrismVertexBuffer[0], prismId, 5)};


//    for(unsigned int faceId = 0; faceId < 5; ++faceId)
//    {
//        ElVisFloat t = MAKE_FLOAT(-1.0);
//        if( IntersectsPrismFace(prismId, faceId, p, origin, W, t) )
//        {
//            if(  rtPotentialIntersection( t ) )
//            {
//                intersectedPrismId = prismId;
//                volumePayload.FoundIntersection = 1;
//                volumePayload.ElementId = prismId;
//                volumePayload.ElementTypeId = 1;
//                volumePayload.IntersectionT = t;
//                rtReportIntersection(0);
//            }
//        }
//    }
}

__device__ __forceinline__ void PrismContainsOrigin(int prismId)
{ 
    WorldPoint origin = MakeFloat3(ray.origin);
    // All planes point out, so each plane needs to return <= 0.
    ElVisFloat p0 = EvaluatePlane(GetPrismPlane(&PrismPlaneBuffer[0], prismId, 0), origin);
    ElVisFloat p1 = EvaluatePlane(GetPrismPlane(&PrismPlaneBuffer[0], prismId, 1), origin);
    ElVisFloat p2 = EvaluatePlane(GetPrismPlane(&PrismPlaneBuffer[0], prismId, 2), origin);
    ElVisFloat p3 = EvaluatePlane(GetPrismPlane(&PrismPlaneBuffer[0], prismId, 3), origin);
    ElVisFloat p4 = EvaluatePlane(GetPrismPlane(&PrismPlaneBuffer[0], prismId, 4), origin);
  

    if( p0 <= MAKE_FLOAT(0.001) && p1 <= MAKE_FLOAT(0.001) && p2 <= MAKE_FLOAT(0.001) && p3 <= MAKE_FLOAT(0.001) && p4 <= MAKE_FLOAT(0.001) )
    {
        // As a final check, make sure the tensor point transformation is in range.
        // This helps fix errors in plane comparisons.
//        TensorPoint tp = TransformPrismWorldToTensor(&PrismVertexBuffer[0], prismId, origin);

//        if( tp.x <= MAKE_FLOAT(-1.01) || tp.x >= MAKE_FLOAT(1.01) ||
//            tp.y <= MAKE_FLOAT(-1.01) || tp.y >= MAKE_FLOAT(1.01) ||
//            tp.z <= MAKE_FLOAT(-1.01) || tp.z >= MAKE_FLOAT(1.01) )
//        {
//            return;
//        }

        if(  rtPotentialIntersection( .1f ) )
        {
            intersectedPrismId = prismId;
            intersectionPointPayload.elementId = prismId;
            intersectionPointPayload.elementType = 1;
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void PrismIntersection(int prismId)
{
    if( ray.ray_type == 1 )
    {
        PrismContainsOrigin(prismId);
    }
    else
    {
        FindRayPrismIntersection(prismId);
    }
}


RT_PROGRAM void PrismBounding (int prismId, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    float3 v0 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 0));
    float3 v1 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 1));
    float3 v2 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 2));
    float3 v3 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 3));
    float3 v4 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 4));
    float3 v5 = ConvertToFloat3(GetPrismVertex(&PrismVertexBuffer[0], prismId, 5));

    aabb->m_min.x = fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_min.y = fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_min.z = fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);
}






#endif
