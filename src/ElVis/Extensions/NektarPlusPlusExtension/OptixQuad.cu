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

#ifndef ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_QUAD_CU
#define ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_QUAD_CU

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

rtBuffer<uint> QuadVertexIndices;
rtBuffer<uint2> QuadModes;

rtBuffer<uint> QuadCoeffMappingDir0;
rtBuffer<uint> QuadCoeffMappingDir1;
rtBuffer<ElVisFloat> QuadMappingCoeffsDir0;
rtBuffer<ElVisFloat> QuadMappingCoeffsDir1;


__device__ __forceinline__ const ElVisFloat4& GetQuadVertex(int id, int vertexId)
{
    unsigned int globalVertex = QuadVertexIndices[id*4 + vertexId];
    ElVisFloat4& result = Vertices[globalVertex];
    return result;
}


RT_PROGRAM void NektarQuadClosestHit()
{
    payload.scalarValue = payload.elementId;
    payload.isValid = true;
}

ELVIS_DEVICE void RefToWorldQuad(const ElVisFloat* u0,
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

        for(int j = 0; j < numModes2; ++j)
        {
            ElVisFloat poly = ModifiedA(j, local.y);
            accum[0] += u0[idx]*poly;
            accum[1] += u1[idx]*poly;
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedA(i, local.x);
        global.x += accum[0]*outerPoly;
        global.y += accum[1]*outerPoly;
    }
    //global.x += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
    //    u0[1];
    //global.y += ModifiedA(1, local.x) * ModifiedB(0, 1, local.y) *
    //    u1[1];
}

ELVIS_DEVICE ElVisFloat RefToWorldQuad_df_dr(const ElVisFloat* u, 
    int numModes1, int numModes2, 
    const ElVisFloat2& local)
{
    //ELVIS_PRINTF("RefToWorldQuad_df_dr Modes (%d, %d)\n", numModes1, numModes2);
    ElVisFloat result = MAKE_FLOAT(0.0);
    int idx = 0;
    for(int i = 0; i < numModes1; ++i)
    {
        ElVisFloat accum = MAKE_FLOAT(0.0);
        for(int j = 0; j < numModes2; ++j)
        {
            ElVisFloat poly = ModifiedA(j, local.y);
            accum += u[idx]*poly;
            //ELVIS_PRINTF("RefToWorldQuad_df_dr Poly (%2.15f) u (%2.15f)\n", poly, u[idx]);
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedAPrime(i, local.x);
        //ELVIS_PRINTF("RefToWorldTriangle_df_dr Outer poly (%2.15f)\n", outerPoly);
        result += accum*outerPoly;
    }
    //result += ModifiedAPrime(1, local.x) * ModifiedB(0, 1, local.y) *
    //    u[1];
    //ELVIS_PRINTF("RefToWorldQuad_df_dr Result (%2.15f)\n", result);
    return result;
}

ELVIS_DEVICE ElVisFloat RefToWorldQuad_df_ds(const ElVisFloat* u,
    int numModes1, int numModes2, 
    const ElVisFloat2& local)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    int idx = 0;
    for(int i = 0; i < numModes1; ++i)
    {
        ElVisFloat accum = MAKE_FLOAT(0.0);
        for(int j = 0; j < numModes2; ++j)
        {
            ElVisFloat poly = ModifiedAPrime(j, local.y);
            accum += u[idx]*poly;
            //ELVIS_PRINTF("RefToWorldTriangle_df_ds poly(%f) accum (%f) u(%f)\n", poly, accum, u[idx]);
            ++idx;
        }

        ElVisFloat outerPoly = ModifiedA(i, local.x);
        result += accum*outerPoly;
        //ELVIS_PRINTF("RefToWorldTriangle_df_ds outerPoly(%f) result (%f)\n", outerPoly, result);
    }
    //result += ModifiedA(1, local.x) * ModifiedBPrime(0, 1, local.y) *
    //    u[1];
    return result;
}

ELVIS_DEVICE ElVisFloat2 NektarQuadWorldPointToReference(int elementId, 
    const ElVisFloat3& intersectionPoint, ElVisFloat2 local)
{
    // Now test the inverse and make sure I can get it right.
    // Start the search in the middle of the element.
    ElVisFloat2 curGlobalPoint = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));

    ElVisFloat J[4];

    ElVisFloat* u0 = &QuadMappingCoeffsDir0[QuadCoeffMappingDir0[elementId]];
    ElVisFloat* u1 = &QuadMappingCoeffsDir1[QuadCoeffMappingDir1[elementId]];
    uint2 modes = QuadModes[elementId];
    ElVisFloat2 global = MakeFloat2(intersectionPoint.x, intersectionPoint.y);
    ELVIS_PRINTF("NektarQuadWorldPointToReference: Target global (%f, %f) \n", global.x, global.y);

    unsigned int numIterations = 0;
    ElVisFloat tolerance = MAKE_FLOAT(1e-5);
    do
    {
        //exp->GetCoord(local, curGlobalPoint);
        RefToWorldQuad(u0, u1, modes.x, modes.y,
            local, curGlobalPoint);
        ElVisFloat2 f;
        f.x = curGlobalPoint.x-global.x;
        f.y = curGlobalPoint.y-global.y;
        ELVIS_PRINTF("NektarQuadWorldPointToReference: Local -> Global (%f, %f) -> (%f, %f)\n", local.x, local.y, curGlobalPoint.x, curGlobalPoint.y);

        J[0] = RefToWorldQuad_df_dr(u0, modes.x, modes.y, local);
        J[1] = RefToWorldQuad_df_ds(u0, modes.x, modes.y, local);
        J[2] = RefToWorldQuad_df_dr(u1, modes.x, modes.y, local);
        J[3] = RefToWorldQuad_df_ds(u1, modes.x, modes.y, local);
     
        ELVIS_PRINTF("NektarQuadWorldPointToReference: J (%2.15f, %2.15f, %2.15f, %2.15f) \n", J[0], J[1], J[2], J[3]);

        ElVisFloat inverse[4];
        ElVisFloat denom = J[0]*J[3] - J[1]*J[2];
        ELVIS_PRINTF("NektarQuadWorldPointToReference: denom %2.15f \n", denom);
        ElVisFloat determinant = MAKE_FLOAT(1.0)/(denom);
        inverse[0] = determinant*J[3];
        inverse[1] = -determinant*J[1];
        inverse[2] = -determinant*J[2];
        inverse[3] = determinant*J[0];
        ELVIS_PRINTF("NektarQuadWorldPointToReference: inverse (%2.15f, %2.15f, %2.15f, %2.15f) \n", inverse[0], inverse[1], inverse[2], inverse[3]);
        double r_adjust = inverse[0]*f.x + inverse[1]*f.y;
        double s_adjust = inverse[2]*f.x + inverse[3]*f.y;

        ELVIS_PRINTF("NektarQuadWorldPointToReference: adjus (%2.15f, %2.15f) \n", r_adjust, s_adjust);


        if( fabsf(r_adjust) < tolerance &&
            fabsf(s_adjust) < tolerance )
        {
            break;
        }

        local.x -= r_adjust;
        local.y -= s_adjust;

        ++numIterations;
    }
    while( numIterations < 50);
    return local;
}

ELVIS_DEVICE bool NektarQuadWorldPointInTriangle(int elementId, 
    const ElVisFloat3& intersectionPoint, const ElVisFloat2& testPoint)
{
    ElVisFloat2 local = NektarQuadWorldPointToReference(elementId, intersectionPoint, testPoint);
    ELVIS_PRINTF("NektarQuadWorldPointInTriangle: (%f, %f)\n", local.x, local.y);
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

ELVIS_DEVICE bool NektarQuadWorldPointInTriangle(int elementId, 
    const ElVisFloat3& intersectionPoint)
{
    ElVisFloat2 p[] = 
    { MakeFloat2(0.0, 0.0)
    ,MakeFloat2(-1.0, -1.0)
    ,MakeFloat2(-1.0, 1.0)
    ,MakeFloat2(1.0, -1.0)
    ,MakeFloat2(1.0, 1.0)
    ,MakeFloat2(-.5, -.5)
    ,MakeFloat2(-.5, .5)
    ,MakeFloat2(.5, -.5)
    ,MakeFloat2(.5, .5) };


    for(int i = 0; i < sizeof(p)/sizeof(ElVisFloat2); ++i)
    {
        if( NektarQuadWorldPointInTriangle(elementId, intersectionPoint, p[i]) )
        {
            return true;
        }
    }
    return false;
}

ELVIS_DEVICE bool Is2DCounterClockwise(const ElVisFloat3& v0, const ElVisFloat3& v1, const ElVisFloat3& v2)
{
    ElVisFloat3 e0 = v1 - v0;
    ElVisFloat3 e1 = v0 - v2;
    ElVisFloat3 n  = cross( e0, e1 );

    ElVisFloat3 e2 = v0 - MakeFloat3(ray.origin);
    ElVisFloat va  = dot( n, e2 );

//    ELVIS_PRINTF("va %2.15f\n", va);
    return (va > 0.0);

}

ELVIS_DEVICE bool Triangle2DIntersection(int primitiveId, const ElVisFloat3& a, 
    const ElVisFloat3& b, const ElVisFloat3& c )
{
    ELVIS_PRINTF("TriangleIntersection (%f, %f, %f), (%f, %f, %f), (%f, %f, %f).\n", a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    ElVisFloat3 v0 = a;
    ElVisFloat3 v1 = b;
    ElVisFloat3 v2 = c;

    if( !Is2DCounterClockwise(a, b, c) )
    {
        v0 = c;
        v2 = a;
    }

    ElVisFloat3 e0 = v1 - v0;
    ElVisFloat3 e1 = v0 - v2;
    ElVisFloat3 n  = cross( e0, e1 );

    ElVisFloat v   = dot( n, MakeFloat3(ray.direction) );
    ElVisFloat r   = MAKE_FLOAT(1.0) / v;

    ElVisFloat3 e2 = v0 - MakeFloat3(ray.origin);
    ElVisFloat va  = dot( n, e2 );
    ElVisFloat t   = r*va;

    if(t < ray.tmax && t > ray.tmin)
    {
        ElVisFloat3 i   = cross( e2, MakeFloat3(ray.direction) );
        ElVisFloat v1   = dot( i, e1 );
        ElVisFloat beta = r*v1;
        if(beta >= MAKE_FLOAT(0.0))
        {
            ElVisFloat v2 = dot( i, e0 );
            ElVisFloat gamma = r*v2;
            if( (v1+v2)*v <= v*v && gamma >= MAKE_FLOAT(0.0) )
            {
                return true;
            }
        }
    }
    return false;
}

RT_PROGRAM void NektarQuadIntersection(int elementId)
{
    // Assume xy plane.
    //if( elementId != 41 ) return;
    ELVIS_PRINTF("NektarQuadIntersection: Element id %d\n", elementId);
    ElVisFloat4 plane = MakeFloat4(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0), MAKE_FLOAT(0.0));
    ElVisFloat t;
    if( FindPlaneIntersection(ray, plane, t) )
    {

        ElVisFloat3 intersectionPoint = MakeFloat3(ray.origin) + t*MakeFloat3(ray.direction);
        ElVisFloat3 v0 = MakeFloat3(GetQuadVertex(elementId, 0));
        ElVisFloat3 v1 = MakeFloat3(GetQuadVertex(elementId, 1));
        ElVisFloat3 v2 = MakeFloat3(GetQuadVertex(elementId, 2));
        ElVisFloat3 v3 = MakeFloat3(GetQuadVertex(elementId, 3));

        bool inTriangle = Triangle2DIntersection(elementId, v0, v1, v2) ||
            Triangle2DIntersection(elementId, v0, v2, v3);
        bool inCurved = NektarQuadWorldPointInTriangle(elementId, intersectionPoint);

        if( inTriangle /*|| inCurved*/)
        {
            if(  rtPotentialIntersection( t ) )
            {
                payload.elementId = elementId;
                payload.elementType = 5;
                payload.IntersectionPoint = intersectionPoint;
                payload.IntersectionT = t;
                payload.Normal = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));
                payload.ReferencePointSet = false;
                rtReportIntersection(0);
            }
        }
    }
}

// Bounding box based on vertices only, padding to deal with curved 
// edges.
RT_PROGRAM void NektarQuadBounding (int id, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;

    const ElVisFloat4& v0 = GetQuadVertex(id, 0);
    const ElVisFloat4& v1 = GetQuadVertex(id, 1);
    const ElVisFloat4& v2 = GetQuadVertex(id, 2);
    const ElVisFloat4& v3 = GetQuadVertex(id, 3);

    aabb->m_min.x = fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x);
    aabb->m_min.y = fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y);
    aabb->m_min.z = fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z)-1.0;

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z)+1.0;
}

#endif
