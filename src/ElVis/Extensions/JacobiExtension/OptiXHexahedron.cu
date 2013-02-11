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

#ifndef ELVIS_EXTENSIONS_JACOBI_EXTENSION_OPTIX_HEXAHEDRON_CU
#define ELVIS_EXTENSIONS_JACOBI_EXTENSION_OPTIX_HEXAHEDRON_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <ElVis/Core/matrix.cu>
#include <optixu/optixu_aabb.h>
//#include "matrix.cu"
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/jacobi.cu>
#include <ElVis/Core/util.cu>
#include <ElVis/Core/OptixVariables.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>

#include <ElVis/Extensions/JacobiExtension/HexahedronCommon.cu>

// The vertices associated with this hex.
rtBuffer<ElVisFloat4> HexVertexBuffer;

// Hexvertex_face_index[i] gives the index for the four 
// vertices associated with face i.
rtBuffer<uint4> Hexvertex_face_index;

// Defines the planes for each hex side.
rtBuffer<ElVisFloat4> HexPlaneBuffer;

rtBuffer<ElVisFloat> HexCoefficients;
rtBuffer<uint> HexCoefficientIndices;

rtBuffer<uint3> HexDegrees;


rtDeclareVariable(int, intersectedHexId, attribute IntersectedHex, );









__device__ __forceinline__ bool IntersectsFace(int hexId, unsigned int faceNumber,
                               ElVisFloat4* p, const ElVisFloat3& origin, const ElVisFloat3& direction,
                               ElVisFloat& t)
{
    uint4 index = Hexvertex_face_index[faceNumber];
    bool result = false;
    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
    {
        result = FindPlaneIntersection(origin, direction, GetPlane(&HexPlaneBuffer[0], hexId, faceNumber), t);
    }   
     
    return result;
}


// Half plane version.
//// Determines if the given ray intersects the given hex.  Returns true if it does, false otherwise.  If an intersection is
//// found, t is the value of the closest intersection.
//__device__ bool HexahedronIntersection(const ElVisFloat3& origin, const ElVisFloat3& direction, int hexId, const ElVisFloat& closestT, ElVisFloat& t)
//{
//    t = closestT;
//    for(int faceId = 0; faceId < 6; ++faceId)
//    {
//        // Check to see if we intersect this face.
//        ElVisFloat plane_t;
//        bool intersectsFace = FindPlaneIntersection(origin, direction, GetPlane(HexPlaneBuffer, hexId, faceId), plane_t);

//        bool testInside = intersectsFace;
//        testInside &= (plane_t < t );
//        if( testInside )
//        {
//            WorldPoint intersectionPoint = origin + plane_t*direction;

//            bool insideOtherFaces = true;
//            for(int insideFaceId = 0; insideFaceId < 6; ++insideFaceId)
//            {
//                if( insideFaceId != faceId )
//                {
//                    ElVisFloat planeVal = EvaluatePlane(GetPlane(HexPlaneBuffer, hexId, insideFaceId), intersectionPoint);
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

__device__ __forceinline__ void FindRayElementIntersection(int hexId)
{
//    // This method causes slow compiles
//    ELVIS_PRINTF("FindRayElementIntersection, ray extents [%f, %f]\n", ray.tmin, ray.tmax);
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
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 0),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 1),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 2),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 3),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 4),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 5),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 6),
//        M3*GetVertex(&HexVertexBuffer[0], hexId, 7) };

//    for(unsigned int faceId = 0; faceId < 6; ++faceId)
//    {
//        ElVisFloat t = MAKE_FLOAT(-1.0);
//        if( IntersectsFace(hexId, faceId, p, origin, W, t) )
//        {
//            if(  rtPotentialIntersection( t ) )
//            {
//                intersectedHexId = hexId;
//                volumePayload.FoundIntersection = 1;
//                volumePayload.ElementId = hexId;
//                volumePayload.ElementTypeId = 0;
//                volumePayload.IntersectionT = t;
//                rtReportIntersection(0);
//            }
//        }
//    }
}


__device__ __forceinline__ void  CheckIfOriginIsInElement(int hexId)
{
    ElVisFloat3 origin = MakeFloat3(ray.origin);
    // All planes point out, so each plane needs to return <= 0.
    ElVisFloat p0 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 0), origin);
    ElVisFloat p1 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 1), origin);
    ElVisFloat p2 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 2), origin);
    ElVisFloat p3 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 3), origin);
    ElVisFloat p4 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 4), origin);
    ElVisFloat p5 = EvaluatePlane(GetPlane(&HexPlaneBuffer[0], hexId, 5), origin);

    

    if( p0 <= MAKE_FLOAT(0.001) && p1 <= MAKE_FLOAT(0.001) && p2 <= MAKE_FLOAT(0.001) && p3 <= MAKE_FLOAT(0.001) && p4 <= MAKE_FLOAT(0.001) && p5 <= MAKE_FLOAT(0.001) )
    {
       
        //ElVis::TensorPoint tp = TransformWorldToTensor(ray.origin);
        //if( tp.x <= -1 || tp.x >= 1 ||
        //    tp.y <= -1 || tp.y >= 1 ||
        //    tp.z <= -1 || tp.z >= 1 )
        //{
        //    return;
        //}

        if(  rtPotentialIntersection( .1 ) )
        {
            intersectedHexId = hexId;
            intersectionPointPayload.elementId = hexId;
            intersectionPointPayload.elementType = 0;
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void HexahedronIntersection(int hexId)
{
    if( ray.ray_type == 1 )
    {
        // Find Element Ray
        CheckIfOriginIsInElement(hexId);
    }
    else
    {
        FindRayElementIntersection(hexId);
    }
}




RT_PROGRAM void hexahedron_bounding (int id, float result[6])
{
    
    optix::Aabb* aabb = (optix::Aabb*)result;
    const ElVisFloat4& v0 = GetVertex(&HexVertexBuffer[0], id, 0);
    const ElVisFloat4& v1 = GetVertex(&HexVertexBuffer[0], id, 1);
    const ElVisFloat4& v2 = GetVertex(&HexVertexBuffer[0], id, 2);
    const ElVisFloat4& v3 = GetVertex(&HexVertexBuffer[0], id, 3);
    const ElVisFloat4& v4 = GetVertex(&HexVertexBuffer[0], id, 4);
    const ElVisFloat4& v5 = GetVertex(&HexVertexBuffer[0], id, 5);
    const ElVisFloat4& v6 = GetVertex(&HexVertexBuffer[0], id, 6);
    const ElVisFloat4& v7 = GetVertex(&HexVertexBuffer[0], id, 7);

    aabb->m_min.x = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    aabb->m_min.y = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    aabb->m_min.z = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);
}




#endif
