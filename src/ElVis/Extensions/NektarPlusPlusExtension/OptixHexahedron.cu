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

// Methods for evaluating arbitrary Nektar++ Hexahedra.
// Phase 1 - Assume planar elements to make sure we're reading all of the data 
// correctly.

#ifndef ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEXAHEDRON_CU
#define ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_HEXAHEDRON_CU

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

#include <SpatialDomains/GeometryShapeType.h>

rtBuffer<uint> HexVertexIndices;
rtBuffer<uint3> NumberOfModes;
rtBuffer<uint4> Hexvertex_face_index;
rtBuffer<ElVisFloat4> HexPlaneBuffer;
rtDeclareVariable(int, intersectedElementId, attribute intersectedElementId, );


__device__ __forceinline__ const ElVisFloat4& GetVertex(int hexId, int vertexId)
{
    unsigned int globalVertex = HexVertexIndices[hexId*8 + vertexId];
    ElVisFloat4& result = Vertices[globalVertex];
    return result;
}

__device__ __forceinline__ const ElVisFloat4& GetPlane(int hexId, int planeId)
{
    ElVisFloat4& result =  HexPlaneBuffer[hexId*8 + planeId];
    rtPrintf("Hex %d, Face %d, Plane (%f, %f, %f, %f)\n", hexId, planeId, result.x, result.y, result.z, result.w);
    return result;
}

__device__ __forceinline__ WorldPoint TransformReferenceToWorld(int hexId, const ReferencePoint& p)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;


    ElVisFloat t1 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t2 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t3 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t4 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t5 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)+t);
    ElVisFloat t6 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)+t);
    ElVisFloat t7 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)+t);
    ElVisFloat t8 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)+t);

    ElVisFloat x = MAKE_FLOAT(.125) * (t1*GetVertex(hexId, 0).x + t2*GetVertex(hexId, 1).x +
        t3*GetVertex(hexId, 2).x + t4*GetVertex(hexId, 3).x +
        t5*GetVertex(hexId, 4).x + t6*GetVertex(hexId, 5).x +
        t7*GetVertex(hexId, 6).x + t8*GetVertex(hexId, 7).x);

    ElVisFloat y = MAKE_FLOAT(.125) * (t1*GetVertex(hexId, 0).y + t2*GetVertex(hexId, 1).y +
        t3*GetVertex(hexId, 2).y + t4*GetVertex(hexId, 3).y +
        t5*GetVertex(hexId, 4).y + t6*GetVertex(hexId, 5).y +
        t7*GetVertex(hexId, 6).y + t8*GetVertex(hexId, 7).y);

    ElVisFloat z = MAKE_FLOAT(.125) * (t1*GetVertex(hexId, 0).z + t2*GetVertex(hexId, 1).z +
        t3*GetVertex(hexId, 2).z + t4*GetVertex(hexId, 3).z +
        t5*GetVertex(hexId, 4).z + t6*GetVertex(hexId, 5).z +
        t7*GetVertex(hexId, 6).z + t8*GetVertex(hexId, 7).z);

    WorldPoint result = MakeFloat3(x, y, z);
    return result;
}




__device__ __forceinline__ void calculateTensorToWorldSpaceMappingJacobian(int hexId, const TensorPoint& p, ElVisFloat* J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = 1.0f-s;
    ElVisFloat t2 = 1.0f-t;
    ElVisFloat t3 = t1*t2;
    ElVisFloat t6 = 1.0f+s;
    ElVisFloat t7 = t6*t2;
    ElVisFloat t10 = 1.0f+t;
    ElVisFloat t11 = t1*t10;
    ElVisFloat t14 = t6*t10;
    ElVisFloat t18 = 1.0f-r;
    ElVisFloat t19 = t18*t2;
    ElVisFloat t21 = 1.0f+r;
    ElVisFloat t22 = t21*t2;
    ElVisFloat t26 = t18*t10;
    ElVisFloat t28 = t21*t10;
    ElVisFloat t33 = t18*t1;
    ElVisFloat t35 = t21*t1;
    ElVisFloat t37 = t21*t6;
    ElVisFloat t39 = t18*t6;

    ElVisFloat v1x = GetVertex(hexId, 0).x;
    ElVisFloat v2x = GetVertex(hexId, 1).x;
    ElVisFloat v3x = GetVertex(hexId, 2).x;
    ElVisFloat v4x = GetVertex(hexId, 3).x;
    ElVisFloat v5x = GetVertex(hexId, 4).x;
    ElVisFloat v6x = GetVertex(hexId, 5).x;
    ElVisFloat v7x = GetVertex(hexId, 6).x;
    ElVisFloat v8x = GetVertex(hexId, 7).x;

    ElVisFloat v1y = GetVertex(hexId, 0).y;
    ElVisFloat v2y = GetVertex(hexId, 1).y;
    ElVisFloat v3y = GetVertex(hexId, 2).y;
    ElVisFloat v4y = GetVertex(hexId, 3).y;
    ElVisFloat v5y = GetVertex(hexId, 4).y;
    ElVisFloat v6y = GetVertex(hexId, 5).y;
    ElVisFloat v7y = GetVertex(hexId, 6).y;
    ElVisFloat v8y = GetVertex(hexId, 7).y;

    ElVisFloat v1z = GetVertex(hexId, 0).z;
    ElVisFloat v2z = GetVertex(hexId, 1).z;
    ElVisFloat v3z = GetVertex(hexId, 2).z;
    ElVisFloat v4z = GetVertex(hexId, 3).z;
    ElVisFloat v5z = GetVertex(hexId, 4).z;
    ElVisFloat v6z = GetVertex(hexId, 5).z;
    ElVisFloat v7z = GetVertex(hexId, 6).z;
    ElVisFloat v8z = GetVertex(hexId, 7).z;

    J[0] = -t3*v1x/8.0f+t3*v2x/8.0f+t7*v3x/8.0f-t7*v4x/8.0f-t11*v5x/8.0f+t11*
        v6x/8.0f+t14*v7x/8.0f-t14*v8x/8.0f;
    J[1] = -t19*v1x/8.0f-t22*v2x/8.0f+t22*v3x/8.0f+t19*v4x/8.0f-t26*v5x/8.0f-
        t28*v6x/8.0f+t28*v7x/8.0f+t26*v8x/8.0f;
    J[2] = -t33*v1x/8.0f-t35*v2x/8.0f-t37*v3x/8.0f-t39*v4x/8.0f+t33*v5x/8.0f+
        t35*v6x/8.0f+t37*v7x/8.0f+t39*v8x/8.0f;
    J[3] = -t3*v1y/8.0f+t3*v2y/8.0f+t7*v3y/8.0f-t7*v4y/8.0f-t11*v5y/8.0f+t11*
        v6y/8.0f+t14*v7y/8.0f-t14*v8y/8.0f;
    J[4] = -t19*v1y/8.0f-t22*v2y/8.0f+t22*v3y/8.0f+t19*v4y/8.0f-t26*v5y/8.0f-
        t28*v6y/8.0f+t28*v7y/8.0f+t26*v8y/8.0f;
    J[5] = -t33*v1y/8.0f-t35*v2y/8.0f-t37*v3y/8.0f-t39*v4y/8.0f+t33*v5y/8.0f+
        t35*v6y/8.0f+t37*v7y/8.0f+t39*v8y/8.0f;
    J[6] = -t3*v1z/8.0f+t3*v2z/8.0f+t7*v3z/8.0f-t7*v4z/8.0f-t11*v5z/8.0f+t11*
        v6z/8.0f+t14*v7z/8.0f-t14*v8z/8.0f;
    J[7] = -t19*v1z/8.0f-t22*v2z/8.0f+t22*v3z/8.0f+t19*v4z/8.0f-t26*v5z/8.0f-
        t28*v6z/8.0f+t28*v7z/8.0f+t26*v8z/8.0f;
    J[8] = -t33*v1z/8.0f-t35*v2z/8.0f-t37*v3z/8.0f-t39*v4z/8.0f+t33*v5z/8.0f+
        t35*v6z/8.0f+t37*v7z/8.0f+t39*v8z/8.0f;
}

__device__ __forceinline__ void calculateInverseJacobian(int hexId, const ReferencePoint& p, ElVis::Matrix<3, 3>& inverse)
{
    ElVisFloat J[9];
    calculateTensorToWorldSpaceMappingJacobian(hexId, p, &J[0]);

    // Now take the inverse.
    ElVisFloat determinant = (-J[0]*J[4]*J[8]+J[0]*J[5]*J[7]+J[3]*J[1]*J[8]-J[3]*J[2]*J[7]-J[6]*J[1]*J[5]+J[6]*J[2]*J[4]);
    inverse[0] = (-J[4]*J[8]+J[5]*J[7])/determinant;
    inverse[1] = -(-J[1]*J[8]+J[2]*J[7])/determinant;
    inverse[2] = -(J[1]*J[5]-J[2]*J[4])/determinant;
    inverse[3] = -(-J[3]*J[8]+J[5]*J[6])/determinant;
    inverse[4] = (-J[0]*J[8]+J[2]*J[6])/determinant;
    inverse[5] = (J[0]*J[5]-J[2]*J[3])/determinant;
    inverse[6] = (-J[3]*J[7]+J[4]*J[6])/determinant;
    inverse[7] = (J[0]*J[7]-J[1]*J[6])/determinant;
    inverse[8] = -(J[0]*J[4]-J[1]*J[3])/determinant;

}


__device__ __forceinline__ ReferencePoint TransformWorldToReference(int hexId, const WorldPoint& p)
{
    int runs = 0;

    ElVisFloat tolerance = MAKE_FLOAT(1e-5);

    ++runs;

    // So we first need an initial guess.  We can probably make this smarter, but
    // for now let's go with 0,0,0.
    ReferencePoint result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVis::Matrix<3, 3> inverse;

    int numIterations = 0;
    const int MAX_ITERATIONS = 100;
    do
    {
        WorldPoint f = TransformReferenceToWorld(hexId, result) - p;
        calculateInverseJacobian(hexId, result, inverse);

        ElVisFloat r_adjust = (inverse[0]*f.x + inverse[1]*f.y + inverse[2]*f.z);
        ElVisFloat s_adjust = (inverse[3]*f.x + inverse[4]*f.y + inverse[5]*f.z);
        ElVisFloat t_adjust = (inverse[6]*f.x + inverse[7]*f.y + inverse[8]*f.z);

        if( fabs(r_adjust) < tolerance &&
            fabs(s_adjust) < tolerance &&
            fabs(t_adjust) < tolerance )
        {
            return result;
        }

        ReferencePoint pointAdjust = MakeFloat3(r_adjust, s_adjust, t_adjust);
        ReferencePoint tempResult = result - pointAdjust;

        // If point adjust is so small it wont' change result then we are done.
        //if( result.x == tempResult.x && result.y == tempResult.y && result.z == tempResult.z )
        //{
        //    return result;
        //}

        result = tempResult;

        // Trial 1 - The odds of this are so small that we probably shouldn't check.
        //WorldPoint inversePoint = transformReferenceToWorld(result);
        //if( p.x == inversePoint.x &&
        //    p.y == inversePoint.y &&
        //    p.z == inversePoint.z  )
        //{            
        //    return result;
        //}

        ++numIterations;
    }
    while( numIterations < MAX_ITERATIONS);

    return result;
}


__device__ __forceinline__ TensorPoint TransformWorldToTensor(int hexId, const WorldPoint& p)
{
    ReferencePoint ref = TransformWorldToReference(hexId, p);
    return ref;
}

__device__ __forceinline__ bool IntersectsFace(int hexId, unsigned int faceNumber,
                               ElVisFloat4* p, const ElVisFloat3& origin, const ElVisFloat3& direction,
                               ElVisFloat& t)
{
    uint4 index = Hexvertex_face_index[faceNumber];
    rtPrintf("Hex %d Face %d Indices (%d, %d, %d, %d)\n", hexId, faceNumber, index.x, index.y, index.z, index.w);
    bool result = false;
    rtPrintf("P0 = (%f, %f, %f, %f)\n", p[index.x].x, p[index.x].y, p[index.x].z, p[index.x].w);
    rtPrintf("P1 = (%f, %f, %f, %f)\n", p[index.y].x, p[index.y].y, p[index.y].z, p[index.y].w);
    rtPrintf("P2 = (%f, %f, %f, %f)\n", p[index.z].x, p[index.z].y, p[index.z].z, p[index.z].w);
    rtPrintf("P3 = (%f, %f, %f, %f)\n", p[index.w].x, p[index.w].y, p[index.w].z, p[index.w].w);
    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
    {
        rtPrintf("Face %d contains origin.\n", faceNumber);
        result = FindPlaneIntersection(origin, direction, GetPlane(hexId, faceNumber), t);
    }   
     
    return result;
}

__device__ __forceinline__ void RayHexahedronIntersection(int hexId)
{
    rtPrintf("Ray/Hex Intersection Test.\n");
    rtPrintf("Origin: (%f, %f, %f) Direction (%f, %f, %f)\n", 
        ray.origin.x, ray.origin.y, ray.origin.z,
        ray.direction.x, ray.direction.y, ray.direction.z);
    ElVisFloat3 origin = MakeFloat3(ray.origin);
    ElVisFloat3 W = MakeFloat3(ray.direction);
    normalize(W);
    ElVisFloat3 U,V;

    GenerateUVWCoordinateSystem(W, U, V);
    rtPrintf("U (%f, %f, %f)\n", U.x, U.y, U.z);
    rtPrintf("V (%f, %f, %f)\n", V.x, V.y, V.z);
    rtPrintf("W (%f, %f, %f)\n", W.x, W.y, W.z);
    // Project each vertex onto the ray's coorindate system
    ElVis::Matrix<4,4> M1;
    M1.getData()[0] = U.x;
    M1.getData()[1] = U.y;
    M1.getData()[2] = U.z;
    M1.getData()[3] = MAKE_FLOAT(0.0);

    M1.getData()[4] = V.x;
    M1.getData()[5] = V.y;
    M1.getData()[6] = V.z;
    M1.getData()[7] = MAKE_FLOAT(0.0);

    M1.getData()[8] = W.x;
    M1.getData()[9] = W.y;
    M1.getData()[10] = W.z;
    M1.getData()[11] = MAKE_FLOAT(0.0);

    M1.getData()[12] = MAKE_FLOAT(0.0);
    M1.getData()[13] = MAKE_FLOAT(0.0);
    M1.getData()[14] = MAKE_FLOAT(0.0);
    M1.getData()[15] = MAKE_FLOAT(1.0);

    ElVis::Matrix<4,4> M2;
    M2.getData()[0] = MAKE_FLOAT(1.0);
    M2.getData()[1] = MAKE_FLOAT(0.0);
    M2.getData()[2] = MAKE_FLOAT(0.0);
    M2.getData()[3] = -origin.x;

    M2.getData()[4] = MAKE_FLOAT(0.0);
    M2.getData()[5] = MAKE_FLOAT(1.0);
    M2.getData()[6] = MAKE_FLOAT(0.0);
    M2.getData()[7] = -origin.y;

    M2.getData()[8] = MAKE_FLOAT(0.0);
    M2.getData()[9] = MAKE_FLOAT(0.0);
    M2.getData()[10] = MAKE_FLOAT(1.0);
    M2.getData()[11] = -origin.z;

    M2.getData()[12] = MAKE_FLOAT(0.0);
    M2.getData()[13] = MAKE_FLOAT(0.0);
    M2.getData()[14] = MAKE_FLOAT(0.0);
    M2.getData()[15] = MAKE_FLOAT(1.0);

    ElVis::Matrix<4,4> M3 = M1*M2;
    // Fill in as p16 of pete's book.

    for(unsigned int i = 0; i < 16; ++i)
    {
        rtPrintf("M1(%d) = %f\n", i, M1[i]);
        rtPrintf("M2(%d) = %f\n", i, M2[i]);
        rtPrintf("M3(%d) = %f\n", i, M3[i]);
    }
    ElVisFloat4 p[] = {
        M3*GetVertex(hexId, 0),
        M3*GetVertex(hexId, 1),
        M3*GetVertex(hexId, 2),
        M3*GetVertex(hexId, 3),
        M3*GetVertex(hexId, 4),
        M3*GetVertex(hexId, 5),
        M3*GetVertex(hexId, 6),
        M3*GetVertex(hexId, 7) };

    for(unsigned int i = 0; i < 8; ++i)
    {
        rtPrintf("P[%d] = (%f, %f, %f, %f)\n", i, p[i].x, p[i].y, p[i].z, p[i].w);
    }

    for(unsigned int faceId = 0; faceId < 6; ++faceId)
    {
        ElVisFloat t = MAKE_FLOAT(-1.0);
        rtPrintf("Testing intersection with face %d.\n", faceId);
        if( IntersectsFace(hexId, faceId, p, origin, W, t) )
        {
            rtPrintf("Found intersection for hex %d, face %d, at %f.\n", hexId, faceId, t);
            if(  rtPotentialIntersection( t ) ) 
            {
                rtPrintf("This is the closest intesection..\n");
                intersectedElementId = hexId;
                volumePayload.FoundIntersection = 1;
                volumePayload.ElementId = hexId;
                volumePayload.ElementTypeId = 0;
                volumePayload.IntersectionT = t;
                rtReportIntersection(0);
            }
        }
    }    
}

__device__ __forceinline__ void NektarHexahedronContainsOriginByCheckingPointMapping(int elementId)
{
    ElVisFloat3 origin = MakeFloat3(ray.origin);
    TensorPoint tp = TransformWorldToTensor(elementId, origin);
    if( tp.x <= -1.01 || tp.x >= 1.01 || 
        tp.y <= -1.01 || tp.y >= 1.01 || 
        tp.z <= -1.01 || tp.z >= 1.01 )
    {
        return;
    }

    if(  rtPotentialIntersection( .1 ) ) 
    {
        intersectedElementId = elementId;
        intersectionPointPayload.elementId = elementId;
        intersectionPointPayload.elementType = Nektar::SpatialDomains::eHexahedron;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void HexahedronIntersection(int elementId)
{
    if( ray.ray_type == 1 )
    {
        NektarHexahedronContainsOriginByCheckingPointMapping(elementId);   
    }
    else
    {
        RayHexahedronIntersection(elementId);
    }
}

RT_PROGRAM void NektarHexahedronBounding (int id, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    const ElVisFloat4& v0 = GetVertex(id, 0);
    const ElVisFloat4& v1 = GetVertex(id, 1);
    const ElVisFloat4& v2 = GetVertex(id, 2);
    const ElVisFloat4& v3 = GetVertex(id, 3);
    const ElVisFloat4& v4 = GetVertex(id, 4);
    const ElVisFloat4& v5 = GetVertex(id, 5);
    const ElVisFloat4& v6 = GetVertex(id, 6);
    const ElVisFloat4& v7 = GetVertex(id, 7);

    aabb->m_min.x = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    aabb->m_min.y = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    aabb->m_min.z = fminf(fminf(fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x), v6.x), v7.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y), v6.y), v7.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z), v6.z), v7.z);
}

__device__ __forceinline__ ElVisFloat EvaluateNektarPlusPlusHexAtTensorPoint(unsigned int elementId, const TensorPoint& p)
{
    uint3 modes = NumberOfModes[elementId];
    uint coefficientIndex = CoefficientOffsets[elementId];

    ElVisFloat result = MAKE_FLOAT(0.0);

    //for(unsigned int i = 0; i < modes.x; ++i)
    for(unsigned int k = 0; k < modes.z; ++k)
    {
        ElVisFloat value_k = ModifiedA(k, p.z);
        for(unsigned int j = 0; j < modes.y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, p.y);
            for(unsigned int i = 0; i < modes.x; ++i)
            //for(unsigned int k = 0; k < modes.z; ++k)
            {
                result += Coefficients[coefficientIndex] * 
                    ModifiedA(i, p.x) *
                    value_j * 
                    value_k;
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir1AtTensorPoint(uint elementId, const T& x, const T& y, const T& z)
{
    T result(MAKE_FLOAT(0.0));

    uint3 modes = NumberOfModes[elementId];
    uint coefficientIndex = CoefficientOffsets[elementId];

    for(unsigned int k = 0; k < modes.z; ++k)
    {
        ElVisFloat value_k = ModifiedAPrime(k, z);
        for(unsigned int j = 0; j < modes.y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, y);
            for(unsigned int i = 0; i < modes.x; ++i)
            {
                result += Coefficients[coefficientIndex] * 
                    ModifiedA(i, x) *
                    value_j * 
                    value_k;
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir2AtTensorPoint(uint elementId, const T& x, const T& y, const T& z)
{
    T result(MAKE_FLOAT(0.0));

    uint3 modes = NumberOfModes[elementId];
    uint coefficientIndex = CoefficientOffsets[elementId];

    for(unsigned int k = 0; k < modes.z; ++k)
    {
        ElVisFloat value_k = ModifiedA(k, z);
        for(unsigned int j = 0; j < modes.y; ++j)
        {
            ElVisFloat value_j = ModifiedAPrime(j, y);
            for(unsigned int i = 0; i < modes.x; ++i)
            {
                result += Coefficients[coefficientIndex] * 
                    ModifiedA(i, x) *
                    value_j * 
                    value_k;
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir3AtTensorPoint(uint elementId, const T& x, const T& y, const T& z)
{
    T result(MAKE_FLOAT(0.0));

    uint3 modes = NumberOfModes[elementId];
    uint coefficientIndex = CoefficientOffsets[elementId];

    for(unsigned int k = 0; k < modes.z; ++k)
    {
        ElVisFloat value_k = ModifiedA(k, z);
        for(unsigned int j = 0; j < modes.y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, y);
            for(unsigned int i = 0; i < modes.x; ++i)
            {
                result += Coefficients[coefficientIndex] * 
                    ModifiedAPrime(i, x) *
                    value_j * 
                    value_k;
                ++coefficientIndex;
            }
        }
    }

    return result;
}


//// The closest hit program.  Assumes intersectedHexId has been populated
//// in an intersection program.
//RT_PROGRAM void NektarEvaluateHexScalarValue()
//{
//    // Evalaute the scalar field at this point.
//    ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;

//    TensorPoint p = TransformWorldToTensor(intersectedElementId, worldSpaceCoordinates);
//    uint3 modes = NumberOfModes[intersectedElementId];
//    uint coefficientIndex = CoefficientIndices[intersectedElementId];

//    ElVisFloat result = MAKE_FLOAT(0.0);

//    //for(unsigned int i = 0; i < modes.x; ++i)
//    for(unsigned int k = 0; k < modes.z; ++k)
//    {
//        ElVisFloat value_k = ModifiedA(k, p.z);
//        for(unsigned int j = 0; j < modes.y; ++j)
//        {
//            ElVisFloat value_j = ModifiedA(j, p.y);
//            for(unsigned int i = 0; i < modes.x; ++i)
//            //for(unsigned int k = 0; k < modes.z; ++k)
//            {
//                result += Coefficients[coefficientIndex] *
//                    ModifiedA(i, p.x) *
//                    value_j *
//                    value_k;
//                ++coefficientIndex;
//            }
//        }
//    }

//    intersectionPointPayload.ScalarValue = result;
//    intersectionPointPayload.elementId = intersectedElementId;
//}

//RT_PROGRAM void NektarEvaluateHexScalarValueArrayVersion()
//{
//    // Evalaute the scalar field at this point.
//    ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;

//    TensorPoint p = TransformWorldToTensor(intersectedElementId, worldSpaceCoordinates);
//    uint3 modes = NumberOfModes[intersectedElementId];
//    uint coefficientIndex = CoefficientIndices[intersectedElementId];

//    ElVisFloat result = MAKE_FLOAT(0.0);

//    ElVisFloat phi_k[15];
//    ElVisFloat phi_j[15];
//    ElVisFloat phi_i[15];

//    ModifiedA(modes.z-1, p.z, phi_k);
//    ModifiedA(modes.y-1, p.y, phi_j);
//    ModifiedA(modes.x-1, p.x, phi_i);

//    for(unsigned int k = 0; k < modes.z; ++k)
//    {
//        for(unsigned int j = 0; j < modes.y; ++j)
//        {
//            for(unsigned int i = 0; i < modes.x; ++i)
//            {
//                result += Coefficients[coefficientIndex] *
//                    phi_k[k] * phi_j[j] * phi_i[i];
//                ++coefficientIndex;
//            }
//        }
//    }

//    intersectionPointPayload.ScalarValue = result;
//    intersectionPointPayload.elementId = intersectedElementId;
//}

#endif 
