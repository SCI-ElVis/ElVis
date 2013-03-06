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

#ifndef ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_TRIANGLE_CU
#define ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_TRIANGLE_CU

#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_aabb.h>

#include <ElVis/Core/matrix.cu>
#include <ElVis/Core/CutSurfacePayloads.cu>
#include <ElVis/Core/typedefs.cu>
#include <ElVis/Core/jacobi.cu>
#include <ElVis/Core/util.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/NektarModel.cu>
#include <ElVis/Extensions/NektarPlusPlusExtension/Expansions.cu>

rtBuffer<uint> TriangleVertexIndices;
rtBuffer<uint2> TriangleModes;

rtBuffer<uint> TriangleGlobalIdMap;
rtBuffer<uint> TriangleCoeffMappingDir0;
rtBuffer<uint> TriangleCoeffMappingDir1;
rtBuffer<ElVisFloat> TriangleMappingCoeffsDir0;
rtBuffer<ElVisFloat> TriangleMappingCoeffsDir1;


// Routines for 2D triangular elements.  We assume that they elements 
// are planar (xy plane), but the edges can be curved.  

__device__ __forceinline__ const ElVisFloat4& GetTriangleVertex(int id, int vertexId)
{
    unsigned int globalVertex = TriangleVertexIndices[id*3 + vertexId];
    ElVisFloat4& result = Vertices[globalVertex];
    return result;
}


RT_PROGRAM void TwoDClosestHitProgram()
{
    uint id = TriangleGlobalIdMap[payload.elementId];
    uint3 modes = GetModes(FieldId, id);
    ElVisFloat* coeffs = GetFieldCoefficients(FieldId, id);
    ElVisFloat coeff1 = coeffs[1];
    ElVisFloat result = MAKE_FLOAT(0.0);
    ElVisFloat3& refPoint = payload.ReferenceIntersectionPoint;

    ELVIS_PRINTF("TwoDClosestHitProgram: Element %d RefPoint (%2.15f, %2.15f)\n", payload.elementId, refPoint.x, refPoint.y);

    for(int i = 0; i < modes.x; ++i)
    {
        for(int j = 0; j < modes.y - i; ++j)
        {
            ELVIS_PRINTF("TwoDClosestHitProgram: Coefficient %2.15f\n", *coeffs);
            ElVisFloat firstTerm = ModifiedA(i, refPoint.x);
            ElVisFloat secondTerm = ModifiedB(i,j, refPoint.y);
            ELVIS_PRINTF("TwoDClosestHitProgram: Term (%d, %d): A (%2.15f) B(%2.15f)\n",
                i, j, firstTerm, secondTerm);
            result += (*coeffs) *
                ModifiedA(i, refPoint.x) *
                ModifiedB(i, j, refPoint.y);
            ++coeffs;
            ELVIS_PRINTF("TwoDClosestHitProgram: Result (%2.15f)\n", result);
        }
    }

    result += ModifiedA(1, refPoint.x) * ModifiedB(0, 1, refPoint.y) *
        coeff1;

    payload.scalarValue = result;
    payload.isValid = true;
}

ELVIS_DEVICE void RefToWorldTriangle(const ElVisFloat* u0,
    const ElVisFloat* u1, 
    int numModes1, int numModes2,
    const ElVisFloat2& local,
    ElVisFloat2& global)
{
    global.x = MAKE_FLOAT(0.0);
    global.y = MAKE_FLOAT(0.0);
    
    int idx = 0;
    for(int i = 0; i < numModes1; ++i)
    {
        ElVisFloat accum[] = {MAKE_FLOAT(0.0), MAKE_FLOAT(0.0)};

        for(int j = 0; j < numModes2-i; ++j)
        {
            ElVisFloat poly = ModifiedB(i, j, local.y);
            accum[0] += u0[idx]*poly;
            accum[1] += u1[idx]*poly;
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedA(i, local.x);
        global.x += accum[0]*outerPoly;
        global.y += accum[1]*outerPoly;
    }
    global.x += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
        u0[1];
    global.y += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
        u1[1];
}

ELVIS_DEVICE ElVisFloat RefToWorldTriangle_df_dr(const ElVisFloat* u, 
    int numModes1, int numModes2, 
    const ElVisFloat2& local)
{
    //ELVIS_PRINTF("RefToWorldTriangle_df_dr Modes (%d, %d)\n", numModes1, numModes2);
    ElVisFloat result = MAKE_FLOAT(0.0);
    int idx = 0;
    for(int i = 0; i < numModes1; ++i)
    {
        ElVisFloat accum = MAKE_FLOAT(0.0);
        for(int j = 0; j < numModes2-i; ++j)
        {
            ElVisFloat poly = ModifiedB(i, j, local.y);
            accum += u[idx]*poly;
            //ELVIS_PRINTF("RefToWorldTriangle_df_dr Poly (%2.15f) u (%2.15f)\n", poly, u[idx]);
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedAPrime(i, local.x);
        //ELVIS_PRINTF("RefToWorldTriangle_df_dr Outer poly (%2.15f)\n", outerPoly);
        result += accum*outerPoly;
    }
    result += ModifiedAPrime(1, local.x) * ModifiedB(0, 1, local.y) *
        u[1];
    //ELVIS_PRINTF("RefToWorldTriangle_df_dr Result (%2.15f)\n", result);
    return result;
}

ELVIS_DEVICE ElVisFloat RefToWorldTriangle_df_ds(const ElVisFloat* u,
    int numModes1, int numModes2, 
    const ElVisFloat2& local)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    int idx = 0;
    for(int i = 0; i < numModes1; ++i)
    {
        ElVisFloat accum = MAKE_FLOAT(0.0);
        for(int j = 0; j < numModes2-i; ++j)
        {
            ElVisFloat poly = ModifiedBPrime(i, j, local.y);
            accum += u[idx]*poly;
            //ELVIS_PRINTF("RefToWorldTriangle_df_ds poly(%f) accum (%f) u(%f)\n", poly, accum, u[idx]);
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedA(i, local.x);
        result += accum*outerPoly;
        //ELVIS_PRINTF("RefToWorldTriangle_df_ds outerPoly(%f) result (%f)\n", outerPoly, result);
    }
    result += ModifiedA(1, local.x) * ModifiedBPrime(0, 1, local.y) *
        u[1];
    return result;
}

ELVIS_DEVICE ElVisFloat2 NektarTriangleWorldPointToReference(int elementId, const ElVisFloat3& intersectionPoint)
{
    // Now test the inverse and make sure I can get it right.
    // Start the search in the middle of the element.
    ElVisFloat2 local = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVisFloat2 curGlobalPoint = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));

    ElVisFloat J[4];

    ElVisFloat* u0 = &TriangleMappingCoeffsDir0[TriangleCoeffMappingDir0[elementId]];
    ElVisFloat* u1 = &TriangleMappingCoeffsDir1[TriangleCoeffMappingDir1[elementId]];
    uint2 modes = TriangleModes[elementId];
    ElVisFloat2 global = MakeFloat2(intersectionPoint.x, intersectionPoint.y);
    //ELVIS_PRINTF("NektarTriangleWorldPointInTriangle: Target global (%f, %f) \n", global.x, global.y);

    unsigned int numIterations = 0;
    ElVisFloat tolerance = MAKE_FLOAT(1e-5);
    do
    {
        //exp->GetCoord(local, curGlobalPoint);
        RefToWorldTriangle(u0, u1, modes.x, modes.y,
            local, curGlobalPoint);
        ElVisFloat2 f;
        f.x = curGlobalPoint.x-global.x;
        f.y = curGlobalPoint.y-global.y;
        //ELVIS_PRINTF("NektarTriangleWorldPointInTriangle: Local -> Global (%f, %f) -> (%f, %f)\n", local.x, local.y, curGlobalPoint.x, curGlobalPoint.y);

        J[0] = RefToWorldTriangle_df_dr(u0, modes.x, modes.y, local);
        J[1] = RefToWorldTriangle_df_ds(u0, modes.x, modes.y, local);
        J[2] = RefToWorldTriangle_df_dr(u1, modes.x, modes.y, local);
        J[3] = RefToWorldTriangle_df_ds(u1, modes.x, modes.y, local);
     
        //ELVIS_PRINTF("NektarTriangleWorldPointInTriangle: J (%2.15f, %2.15f, %2.15f, %2.15f) \n", J[0], J[1], J[2], J[3]);

        ElVisFloat inverse[4];
        ElVisFloat denom = J[0]*J[3] - J[1]*J[2];
        //ELVIS_PRINTF("NektarTriangleWorldPointInTriangle: denom %2.15f \n", denom);
        ElVisFloat determinant = MAKE_FLOAT(1.0)/(denom);
        inverse[0] = determinant*J[3];
        inverse[1] = -determinant*J[1];
        inverse[2] = -determinant*J[2];
        inverse[3] = determinant*J[0];

        double r_adjust = inverse[0]*f.x + inverse[1]*f.y;
        double s_adjust = inverse[2]*f.x + inverse[3]*f.y;

        if( fabsf(r_adjust) < tolerance &&
            fabsf(s_adjust) < tolerance )
        {
            break;
        }

        local.x -= r_adjust;
        local.y -= s_adjust;

        ++numIterations;
    }
    while( numIterations < 10);
    return local;
}

ELVIS_DEVICE bool NektarTriangleWorldPointInTriangle(int elementId, const ElVisFloat3& intersectionPoint)
{
    ElVisFloat2 local = NektarTriangleWorldPointToReference(elementId, intersectionPoint);
    //ELVIS_PRINTF("NektarTriangleWorldPointInTriangle: (%f, %f)\n", local.x, local.y);
    if( local.x < -1.01 || local.x > 1.01 ||
        local.y < -1.01 || local.y > 1.01 )
    {
        return false;
    }
    else
    {
        return true;
    }
}


//
//ELVIS_DEVICE bool Is2DCounterClockwise(const ElVisFloat3& v0, const ElVisFloat3& v1, const ElVisFloat3& v2)
//{
//    ElVisFloat3 e0 = v1 - v0;
//    ElVisFloat3 e1 = v0 - v2;
//    ElVisFloat3 n  = cross( e0, e1 );
//
//    ElVisFloat3 e2 = v0 - MakeFloat3(ray.origin);
//    ElVisFloat va  = dot( n, e2 );
//
////    ELVIS_PRINTF("va %2.15f\n", va);
//    return (va > 0.0);
//
//}
//
//ELVIS_DEVICE void Triangle2DIntersection(int primitiveId, const ElVisFloat3& a, 
//    const ElVisFloat3& b, const ElVisFloat3& c )
//{
//    ELVIS_PRINTF("TriangleIntersection (%f, %f, %f), (%f, %f, %f), (%f, %f, %f).\n", a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
//    ElVisFloat3 v0 = a;
//    ElVisFloat3 v1 = b;
//    ElVisFloat3 v2 = c;
//
//    if( !Is2DCounterClockwise(a, b, c) )
//    {
//        v0 = c;
//        v2 = a;
//    }
//
//    ElVisFloat3 e0 = v1 - v0;
//    ElVisFloat3 e1 = v0 - v2;
//    ElVisFloat3 n  = cross( e0, e1 );
//
//    ElVisFloat v   = dot( n, MakeFloat3(ray.direction) );
//    ElVisFloat r   = MAKE_FLOAT(1.0) / v;
//
//    ElVisFloat3 e2 = v0 - MakeFloat3(ray.origin);
//    ElVisFloat va  = dot( n, e2 );
//    ElVisFloat t   = r*va;
//
//    if(t < ray.tmax && t > ray.tmin)
//    {
//        ElVisFloat3 i   = cross( e2, MakeFloat3(ray.direction) );
//        ElVisFloat v1   = dot( i, e1 );
//        ElVisFloat beta = r*v1;
//        if(beta >= MAKE_FLOAT(0.0))
//        {
//            ElVisFloat v2 = dot( i, e0 );
//            ElVisFloat gamma = r*v2;
//            if( (v1+v2)*v <= v*v && gamma >= MAKE_FLOAT(0.0) )
//            {
//                if(  rtPotentialIntersection( t ) )
//                {
//                payload.elementId = primitiveId;
//                payload.elementType = 4;
//                payload.IntersectionPoint = ray.origin + t*ray.direction;
//                payload.IntersectionT = t;
//                payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));
//                payload.ReferencePointSet = false;
//                rtReportIntersection(0);
//                }
//            }
//        }
//    }
//}


RT_PROGRAM void NektarTriangleIntersection(int elementId)
{
    //Triangle2DIntersection(elementId, MakeFloat3(GetTriangleVertex(elementId, 0)), 
    //    MakeFloat3(GetTriangleVertex(elementId, 1)), 
    //    MakeFloat3(GetTriangleVertex(elementId, 2)));
    ELVIS_PRINTF("NektarTriangleIntersection: Element id %d\n", elementId);
    // Assume xy plane.
    ElVisFloat4 plane = MakeFloat4(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0), MAKE_FLOAT(0.0));
    ElVisFloat t;
    if( FindPlaneIntersection(ray, plane, t) )
    {
        ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + t*MakeFloat3(ray.direction);
        ElVisFloat2 refPoint = NektarTriangleWorldPointToReference(elementId, intersectionPoint);
        if( refPoint.x < 1.01 && refPoint.x > -1.01 &&
            refPoint.y < 1.01 && refPoint.y > -1.01 )
        {
            if(  rtPotentialIntersection( t ) )
            {
                payload.elementId = elementId;
                payload.elementType = 4;
                payload.IntersectionPoint = intersectionPoint;
                payload.IntersectionT = t;
                payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));
                payload.ReferencePointSet = true;
                payload.ReferenceIntersectionPoint.x = refPoint.x;
                payload.ReferenceIntersectionPoint.y = refPoint.y;
                rtReportIntersection(0);
            }
        }
    }
}

// Bounding box based on vertices only, padding to deal with curved 
// edges.
RT_PROGRAM void NektarTriangleBounding (int id, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;

    const ElVisFloat4& v0 = GetTriangleVertex(id, 0);
    const ElVisFloat4& v1 = GetTriangleVertex(id, 1);
    const ElVisFloat4& v2 = GetTriangleVertex(id, 2);

    aabb->m_min.x = fminf(fminf(v0.x, v1.x), v2.x);
    aabb->m_min.y = fminf(fminf(v0.y, v1.y), v2.y);
    aabb->m_min.z = fminf(fminf(v0.z, v1.z), v2.z)-1.0;

    aabb->m_max.x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    aabb->m_max.y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    aabb->m_max.z = fmaxf(fmaxf(v0.z, v1.z), v2.z)+1.0;
}

#endif
