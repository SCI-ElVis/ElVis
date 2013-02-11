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

#ifndef ELVIS_JACOBI_EXTENSION_HEXAHEDRON_CU
#define ELVIS_JACOBI_EXTENSION_HEXAHEDRON_CU

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

// The vertices associated with this hex.
rtBuffer<ElVisFloat4> HexVertexBuffer;

// Hexvertex_face_index[i] gives the index for the four 
// vertices associated with face i.
rtBuffer<uint4> Hexvertex_face_index;

// Defines the planes for each hex side.
rtBuffer<float4> HexPlaneBuffer;

rtBuffer<ElVisFloat> HexCoefficients;
rtBuffer<uint> HexCoefficientIndices;

rtBuffer<uint3> HexDegrees;


rtDeclareVariable(int, intersectedHexId, attribute IntersectedHex, );


__device__ __forceinline__ const ElVisFloat4& GetVertex(int hexId, int vertexId)
{
    return HexVertexBuffer[hexId*8 + vertexId];
}

__device__ __forceinline__ const float4& GetPlane(int hexId, int planeId)
{
    return HexPlaneBuffer[hexId*8 + planeId];
}


__device__ __forceinline__ void TransformReferenceToWorld(int hexId, const ReferencePoint& p,
                                                          WorldPoint& result)
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


    result.x = x;
    result.y = y;
    result.z = z;
}




////////////////////////////////////////////////////////////////////////
// Array Versions
////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ void calculateTensorToWorldSpaceMappingJacobian(int hexId, 
                                                                           const TensorPoint& p, 
                                                                           ElVisFloat* J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    
    ElVisFloat t1 = MAKE_FLOAT(1.0)-s;
    ElVisFloat t2 = MAKE_FLOAT(1.0)-t;
    ElVisFloat t6 = MAKE_FLOAT(1.0)+s;
    ElVisFloat t10 = MAKE_FLOAT(1.0)+t;
    ElVisFloat t18 = MAKE_FLOAT(1.0)-r;
    ElVisFloat t21 = MAKE_FLOAT(1.0)+r;

    ElVisFloat t3 = t1*t2*MAKE_FLOAT(.125);
    ElVisFloat t11 = t1*t10*MAKE_FLOAT(.125);
    ElVisFloat t33 = t18*t1*MAKE_FLOAT(.125);
    ElVisFloat t35 = t21*t1*MAKE_FLOAT(.125);

    ElVisFloat t7 = t6*t2*MAKE_FLOAT(.125);
    ElVisFloat t19 = t18*t2*MAKE_FLOAT(.125);
    ElVisFloat t22 = t21*t2*MAKE_FLOAT(.125);
    
    ElVisFloat t37 = t21*t6*MAKE_FLOAT(.125);
    ElVisFloat t39 = t18*t6*MAKE_FLOAT(.125);
    ElVisFloat t14 = t6*t10*MAKE_FLOAT(.125);
    ElVisFloat t26 = t18*t10*MAKE_FLOAT(.125);
    ElVisFloat t28 = t21*t10*MAKE_FLOAT(.125);
    
    

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

    J[0] = -t3*v1x+t3*v2x+t7*v3x-t7*v4x-t11*v5x+t11*
        v6x+t14*v7x-t14*v8x;
    J[1] = -t19*v1x-t22*v2x+t22*v3x+t19*v4x-t26*v5x-
        t28*v6x+t28*v7x+t26*v8x;
    J[2] = -t33*v1x-t35*v2x-t37*v3x-t39*v4x+t33*v5x+
        t35*v6x+t37*v7x+t39*v8x;
    J[3] = -t3*v1y+t3*v2y+t7*v3y-t7*v4y-t11*v5y+t11*
        v6y+t14*v7y-t14*v8y;
    J[4] = -t19*v1y-t22*v2y+t22*v3y+t19*v4y-t26*v5y-
        t28*v6y+t28*v7y+t26*v8y;
    J[5] = -t33*v1y-t35*v2y-t37*v3y-t39*v4y+t33*v5y+
        t35*v6y+t37*v7y+t39*v8y;
    J[6] = -t3*v1z+t3*v2z+t7*v3z-t7*v4z-t11*v5z+t11*
        v6z+t14*v7z-t14*v8z;
    J[7] = -t19*v1z-t22*v2z+t22*v3z+t19*v4z-t26*v5z-
        t28*v6z+t28*v7z+t26*v8z;
    J[8] = -t33*v1z-t35*v2z-t37*v3z-t39*v4z+t33*v5z+
        t35*v6z+t37*v7z+t39*v8z;
}

__device__ __forceinline__ void calculateInverseJacobian(int hexId, const ReferencePoint& p, 
                                                         ElVisFloat* inverse)
{
    //rtPrintf("calculateInverseJacobian");
    ElVisFloat J[16];
    calculateTensorToWorldSpaceMappingJacobian(hexId, p, &J[0]);

    // Now take the inverse.
    ElVisFloat determinant = (-J[0]*J[4]*J[8]+J[0]*J[5]*J[7]+J[3]*J[1]*J[8]-J[3]*J[2]*J[7]-J[6]*J[1]*J[5]+J[6]*J[2]*J[4]);
    ElVisFloat i = MAKE_FLOAT(1.0)/determinant;
    inverse[0] = (-J[4]*J[8]+J[5]*J[7])*i;
    inverse[1] = -(-J[1]*J[8]+J[2]*J[7])*i;
    inverse[2] = -(J[1]*J[5]-J[2]*J[4])*i;
    inverse[3] = -(-J[3]*J[8]+J[5]*J[6])*i;
    inverse[4] = (-J[0]*J[8]+J[2]*J[6])*i;
    inverse[5] = (J[0]*J[5]-J[2]*J[3])*i;
    inverse[6] = (-J[3]*J[7]+J[4]*J[6])*i;
    inverse[7] = (J[0]*J[7]-J[1]*J[6])*i;
    inverse[8] = -(J[0]*J[4]-J[1]*J[3])*i;

}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ ReferencePoint TransformWorldToReference(int hexId, const WorldPoint& p)
{
    //int exact = 0;
    //int runs = 0;
    //int iteration = 0;
    //int adjust = 0;

    ElVisFloat tolerance = MAKE_FLOAT(1e-5);

    //++runs;

    // So we first need an initial guess.  We can probably make this smarter, but
    // for now let's go with 0,0,0.
    ReferencePoint result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    
    //ElVisFloat AlignmentTestMatrix[64];
    
    int numIterations = 0;
    const int MAX_ITERATIONS = 10;
    do
    {
        ElVisFloat inverse[16];
        //ElVis::Matrix<3, 3> inverse;
        
        WorldPoint f;
        TransformReferenceToWorld(hexId, result, f);
        f.x -= p.x;
        f.y -= p.y;
        f.z -= p.z;

        calculateInverseJacobian(hexId, result, &inverse[0]);

        ElVisFloat r_adjust = (inverse[0]*f.x + inverse[1]*f.y + inverse[2]*f.z);
        ElVisFloat s_adjust = (inverse[3]*f.x + inverse[4]*f.y + inverse[5]*f.z);
        ElVisFloat t_adjust = (inverse[6]*f.x + inverse[7]*f.y + inverse[8]*f.z);

        bool test = fabsf(r_adjust) < tolerance;
        test &= fabsf(s_adjust) < tolerance;
        test &= fabsf(t_adjust) < tolerance;
        if( test ) return result;

        //ReferencePoint pointAdjust = MakeFloat3(r_adjust, s_adjust, t_adjust);
        //ReferencePoint tempResult = result - pointAdjust;

        // If point adjust is so small it wont' change result then we are done.
        //if( result.x == tempResult.x && result.y == tempResult.y && result.z == tempResult.z )
        //{
        //    return result;
        //}

        //result = tempResult;
        result.x -= r_adjust;
        result.y -= s_adjust;
        result.z -= t_adjust;

        // Trial 1 - The odds of this are so small that we probably shouldn't check.
        //ElVis::WorldPoint inversePoint = transformReferenceToWorld(result);
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
    bool result = false;
    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
    {
        result = FindPlaneIntersection(origin, direction, GetPlane(hexId, faceNumber), t);
    }   
     
    return result;
}


RT_PROGRAM void HexahedronIntersection(int hexId)
{
    ElVisFloat3 origin = MakeFloat3(ray.origin);
    ElVisFloat3 W = MakeFloat3(ray.direction);
    normalize(W);
    ElVisFloat3 U,V;
    GenerateUVWCoordinateSystem(W, U, V);
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

    ElVisFloat4 p[] = {
        M3*GetVertex(hexId, 0),
        M3*GetVertex(hexId, 1),
        M3*GetVertex(hexId, 2),
        M3*GetVertex(hexId, 3),
        M3*GetVertex(hexId, 4),
        M3*GetVertex(hexId, 5),
        M3*GetVertex(hexId, 6),
        M3*GetVertex(hexId, 7) };

    //if( launch_index.x == 755 && launch_index.y == 270 )
    //{
    //    rtPrintf("Testing Hex %d\n", hexId);
    //    rtPrintf("Ray Origin (%e, %e, %e)\n", origin.x, origin.y, origin.z);
    //    rtPrintf("Ray Direction (%e, %e, %e)\n", W.x, W.y, W.z);
    //    
    //    for(unsigned int i = 0; i < 8; ++i)
    //    {
    //        rtPrintf("V[%d] = (%e, %e, %e)\n", i,GetVertex(hexId, i).x, GetVertex(hexId, i).y, GetVertex(hexId, i).z);
    //    }

    //    rtPrintf("U (%e, %e, %e), V (%e, %e, %e), W (%e, %e, %e)\n", 
    //        U.x, U.y, U.z, V.x, V.y, V.z, W.x, W.y, W.z);
    //    for(unsigned int i = 0; i < 8; ++i)
    //    {
    //        rtPrintf("P[%d] = (%e, %e, %e)\n", i, p[i].x, p[i].y, p[i].z);
    //    }
    //}

    for(unsigned int faceId = 0; faceId < 6; ++faceId)
    {
        ElVisFloat t = MAKE_FLOAT(-1.0);
        if( IntersectsFace(hexId, faceId, p, origin, W, t) )
        {
            //if( launch_index.x == 755 && launch_index.y == 270 )
            //{
            //    rtPrintf("Intersected face %d at t value %e\n", faceId, t);
            //}
            if(  rtPotentialIntersection( t ) ) 
            {
                //if( launch_index.x == 755 && launch_index.y == 270 )
                //{
                //    rtPrintf("This is closest \n");
                //}
                intersectedHexId = hexId;
                volumePayload.FoundIntersection = 1;
                volumePayload.ElementId = hexId;
                volumePayload.ElementTypeId = 0;
                volumePayload.IntersectionT = t;
                rtReportIntersection(0);
            }
        }
    }    
}




// This is the one being used.
RT_PROGRAM void HexahedronContainsOriginByCheckingPoint(int hexId)
{
    ElVisFloat3 origin = MakeFloat3(ray.origin);
    // All planes point out, so each plane needs to return <= 0.
    ElVisFloat p0 = EvaluatePlane(GetPlane(hexId, 0), origin);
    ElVisFloat p1 = EvaluatePlane(GetPlane(hexId, 1), origin);
    ElVisFloat p2 = EvaluatePlane(GetPlane(hexId, 2), origin);
    ElVisFloat p3 = EvaluatePlane(GetPlane(hexId, 3), origin);
    ElVisFloat p4 = EvaluatePlane(GetPlane(hexId, 4), origin);
    ElVisFloat p5 = EvaluatePlane(GetPlane(hexId, 5), origin);

    

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
            rtReportIntersection(0);
        }
    }
    
    //return;

    //optix::Matrix<2, 3> transformationMatrix;

    //float3* u = (float3*)(transformationMatrix.getData());
    //float3* v = (float3*)(transformationMatrix.getData() + 3);
    //GenerateUVWCoordinateSystem(ray.direction, *u, *v);
    //

    //ElVis::WorldPoint o = ray.origin;
    //ElVis::WorldPoint p[4];
    //uint4 vertex_index = Hexvertex_face_index[hexId];
    //p[0] = make_float3(GetVertex(hexId, vertex_index.x));
    //p[1] = make_float3(GetVertex(hexId, vertex_index.y));
    //p[2] = make_float3(GetVertex(hexId, vertex_index.z));
    //p[3] = make_float3(GetVertex(hexId, vertex_index.w));

    //for(int i = 0; i < 4; i++)
    //{
    //    float tx = p[i].x - o.x;
    //    float ty = p[i].y-o.y;
    //    float tz = p[i].z-o.z;
    //    p[i].x = (*u).x*tx + (*u).y*ty + (*u).z*tz;
    //    p[i].y = (*v).x*tx + (*v).y*ty + (*v).z*tz;
    //    p[i].z = 0.0f;
    //    // Don't worry about the w component.  We want to project onto
    //    // the uv plane so we'll set it to 0.
    //}

    //if( ContainsOrigin(p[0], p[1], p[2], p[3]) )
    //{
    //    ElVis::WorldPoint w = ray.origin;
    //    ElVis::TensorPoint tp = TransformWorldToTensor(hexId, w);
    //    if( tp.x < -1 || tp.x > 1 || 
    //        tp.y < -1 || tp.y > 1 || 
    //        tp.z < -1 || tp.z > 1 )
    //    {
    //        return;
    //    }

    //    float t;
    //    FindPlaneIntersection(ray, plane_buffer[primIdx], t);
    //    if(  rtPotentialIntersection( t ) ) 
    //    {
    //        intersectedHexId = hexId;
    //        rtReportIntersection(0);
    //    }
    //}
}

//// An intersection program that only reports an intersection if the ray origin is inside the element.
//// This isn't working that well, and is very difficult to debug.
//RT_PROGRAM void HexahedronContainsOrigin(int primIdx)
//{
//    // First, see if there is an intersection with the current element and face.
//    // If there is, try it again with any other face.  If both intersect then we 
//    float t1;
//    bool firstTest = IntersectsFace(primIdx, ray.origin, ray.direction, t1);
//    
//
//    if( !firstTest ) return;
//    
//    int nextFace = primIdx - 1;
//    if( nextFace < 0 ) nextFace = primIdx + 1;
//
//    uint4 vertex_index = Hexvertex_face_index[nextFace];
//    float3 p0 = make_float3(vertex_buffer[vertex_index.x]);
//    float3 p2 = make_float3(vertex_buffer[vertex_index.z]);
//    float3 target = p2 + (p0 - p2)/2.0f;
//
//    float3 direction = target - ray.origin;
//
//    float t2;
//    bool secondTest = IntersectsFace(nextFace, ray.origin, direction, t2); 
//    if( !secondTest ) return;
//
//    if(  rtPotentialIntersection( t1 ) ) 
//    {
//        rtReportIntersection(0);
//    }
//}

//RT_PROGRAM void hexahedron_intersect( int primIdx )
//{
//    // primIdx will be 0-5, one for each face.
//
//    // Create the (u,v,w) coordinate system for the ray.
//    
//    // Create a local coordinate system based on the ray.  The viewing 
//    // direction is w, and u and v are arbitrary.  Since we'll be projecting
//    // everything into the u,v plane, it turns out we don't need w, so the 
//    // creation method doesn't need to return w.
//    //
//    // Then generate the following modified coordinate transformation matrix, 
//    // which transforms an (x,y,z) point into (u,v,w) space and projects it 
//    // onto the (u,v) plane.
//    // [ ux  uy  uz ]
//    // [ vx  vy  vz ]
//    optix::Matrix<2, 3> transformationMatrix;
//
//    float3* u = (float3*)(transformationMatrix.getData());
//    float3* v = (float3*)(transformationMatrix.getData() + 3);
//    GenerateUVWCoordinateSystem(ray.direction, *u, *v);
//    
//    // Next, load the vertices associated with this face.
//    //uint4 vertex_index = Hexvertex_face_index[primIdx];
//    //optix::Matrix<3, 4> vertexMatrix;
//    //SetColumn(vertexMatrix, 0, vertex_buffer[vertex_index.x]);
//    //SetColumn(vertexMatrix, 1, vertex_buffer[vertex_index.y]);
//    //SetColumn(vertexMatrix, 2, vertex_buffer[vertex_index.z]);
//    //SetColumn(vertexMatrix, 3, vertex_buffer[vertex_index.w]);
//
//    //optix::Matrix<2, 4> transformedVertices = transformationMatrix*vertexMatrix;
//
//    //float2 p0 = make_float2(transformedVertices.getRow(0));
//    //float2 p1 = make_float2(transformedVertices.getRow(1));
//    //float2 p2 = make_float2(transformedVertices.getRow(2));
//    //float2 p3 = make_float2(transformedVertices.getRow(3));
//
//    ElVis::WorldPoint o = ray.origin;
//    ElVis::WorldPoint p[4];
//    uint4 vertex_index = Hexvertex_face_index[primIdx];
//    p[0] = make_float3(vertex_buffer[vertex_index.x]);
//    p[1] = make_float3(vertex_buffer[vertex_index.y]);
//    p[2] = make_float3(vertex_buffer[vertex_index.z]);
//    p[3] = make_float3(vertex_buffer[vertex_index.w]);
//
//    for(int i = 0; i < 4; i++)
//    {
//        float tx = p[i].x - o.x;
//        float ty = p[i].y-o.y;
//        float tz = p[i].z-o.z;
//        p[i].x = (*u).x*tx + (*u).y*ty + (*u).z*tz;
//        p[i].y = (*v).x*tx + (*v).y*ty + (*v).z*tz;
//        p[i].z = 0.0f;
//        // Don't worry about the w component.  We want to project onto
//        // the uv plane so we'll set it to 0.
//    }
//
//    if( ContainsOrigin(p[0], p[1], p[2], p[3]) )
//    {
//        float t;
//        FindPlaneIntersection(ray, plane_buffer[primIdx], t);
//        if(  rtPotentialIntersection( t ) ) 
//        {
//            rtReportIntersection(0);
//        }
//    }
//}


RT_PROGRAM void hexahedron_bounding (int id, float result[6])
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


template<typename ElementType, typename PointType>
__device__ __forceinline__ ElementType EvaluateHexFieldAtTensorPoint(unsigned int hexId, const PointType& p)
{
    //rtPrintf("EvaluateHexFieldAtTensorPoint\n");
    uint3 degree = HexDegrees[hexId];
    
    ElementType result(MAKE_FLOAT(0.0));

    ElementType phi_k[8];
    ElementType phi_j[8];
    ElementType phi_i[8];

    ElVis::OrthoPoly::P(degree.x, 0, 0, p.x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, p.y, phi_j);
    ElVis::OrthoPoly::P(degree.z, 0, 0, p.z, phi_k);

    uint coefficientIndex = HexCoefficientIndices[hexId];

    //rtPrintf("Evaluating Hex %d at point (%f, %f, %f)\n", hexId, p.x, p.y, p.z);
    //unsigned int dx = degree.x+1;
    //unsigned int dy = degree.y+1;
    //unsigned int dz = degree.z+1;
    //unsigned int limit = dx*dy*dz;

    //unsigned int i = 0;
    //unsigned int j = 0;
    //unsigned int k = 0;
    //for(unsigned int coefficientIndex = 0; coefficientIndex < limit; ++coefficientIndex)
    //{
    //    result += HexCoefficients[coefficientIndex] *
    //        phi_i[i] * phi_j[j] * phi_k[k];
    //
    //    // Should be save and have no divergence.
    //    k += 1;
    //    if( k > dz )
    //    {
    //        k = 0;
    //        j += 1;
    //        if( j > dy )
    //        {
    //            j = 0;
    //            i += 1;
    //        }
    //    }
    //
    //}

    //for(unsigned int i = 0; i <= 6; ++i)
    //{
    //    for(unsigned int j = 0; j <= 6; ++j)
    //    {
    //        for(unsigned int k = 0; k <= 6; ++k)
    //        {
    //            bool test = i <= degree.x;
    //            test &= j <= degree.y;
    //            test &= k <= degree.z;
    //            if( test )
    //            {
    //                result += HexCoefficients[coefficientIndex] *
    //                    phi_i[i] * phi_j[j] * phi_k[k];
    //                ++coefficientIndex;
    //            }
    //        }
    //    }
    //}

//    int dx = degree.x;
//    int dy = degree.y;
//    int dz = degree.z;
//    for(int i = 0; i <= dx; ++i)
//    {
//        ElVisFloat level1 = 0.0;
//        for(int j = 0; j <= dy; ++j)
//        {
//            ElVisFloat level2 = 0.0;
//            for(int k = 0; k <= dz; ++k)
//            {
//                level2 += HexCoefficients[coefficientIndex] * phi_k[k];
//                //   //* phi_i[i] * phi_j[j] * phi_k[k];
//                ++coefficientIndex;
//            }
//            level1 += level2*phi_j[j];
//        }
//        result += level1*phi_i[i];
//    }


    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                rtPrintf("Evaluating term %d with coeff %f, phi_i %f, phi_j %f, and phi_k %f\n",
                    coefficientIndex, HexCoefficients[coefficientIndex], phi_i[i], phi_j[j], phi_k[k]);
                result += HexCoefficients[coefficientIndex]
                    * phi_i[i] * phi_j[j] * phi_k[k];
                ++coefficientIndex;
            }
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluateHexFieldAtWorldPoint(unsigned int hexId, const WorldPoint& worldPoint, TensorPoint& tensorPoint)
{
    tensorPoint = TransformWorldToTensor(hexId, worldPoint);
    return EvaluateHexFieldAtTensorPoint<ElVisFloat>(hexId, tensorPoint);
}

__device__ __forceinline__ ElVisFloat EvaluateHexFieldAtWorldPoint(unsigned int hexId, const WorldPoint& worldPoint)
{
    TensorPoint tensorPoint = TransformWorldToTensor(hexId, worldPoint);
    return EvaluateHexFieldAtTensorPoint<ElVisFloat>(hexId, tensorPoint);
}

// The closest hit program.  Assumes intersectedHexId has been populated 
// in an intersection program.
RT_PROGRAM void EvaluateHexScalarValue()
{
    // Evalaute the scalar field at this point.
    intersectionPointPayload.found = 1;
    
    ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;
    TensorPoint p = TransformWorldToTensor(intersectedHexId, worldSpaceCoordinates);
    uint3 degree = HexDegrees[intersectedHexId];

    ElVisFloat result = MAKE_FLOAT(0.0);

    uint coefficientIndex = HexCoefficientIndices[intersectedHexId];
    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                result += HexCoefficients[coefficientIndex] *
                    ElVis::OrthoPoly::P(i, 0, 0, p.x)*
                    ElVis::OrthoPoly::P(j, 0, 0, p.y)*
                    ElVis::OrthoPoly::P(k, 0, 0, p.z);
                ++coefficientIndex;
            }
        }
    }
    
    intersectionPointPayload.ScalarValue = result;
    intersectionPointPayload.elementId = intersectedHexId;
    intersectionPointPayload.elementType = 0;
    intersectionPointPayload.ReferenceIntersectionPoint = p;
    intersectionPointPayload.found = 1;
}

// The closest hit program.  Assumes intersectedHexId has been populated 
// in an intersection program.
RT_PROGRAM void EvaluateHexScalarValueArrayVersion()
{
    // Evalaute the scalar field at this point.
    intersectionPointPayload.found = 1;
    ElVisFloat result = EvaluateHexFieldAtWorldPoint(intersectedHexId, intersectionPointPayload.IntersectionPoint,
        intersectionPointPayload.ReferenceIntersectionPoint);

    //ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;
    //TensorPoint p = TransformWorldToTensor(intersectedHexId, worldSpaceCoordinates);
    //uint3 degree = HexDegrees[intersectedHexId];

    //ElVisFloat result = MAKE_FLOAT(0.0);

    //ElVisFloat phi_k[15];
    //ElVisFloat phi_j[15];
    //ElVisFloat phi_i[15];

    //ElVis::OrthoPoly::P(degree.x, 0, 0, p.x, phi_i);
    //ElVis::OrthoPoly::P(degree.y, 0, 0, p.y, phi_j);
    //ElVis::OrthoPoly::P(degree.z, 0, 0, p.z, phi_k);

    //uint coefficientIndex = HexCoefficientIndices[intersectedHexId];
    //for(unsigned int i = 0; i <= degree.x; ++i)
    //{
    //    for(unsigned int j = 0; j <= degree.y; ++j)
    //    {
    //        for(unsigned int k = 0; k <= degree.z; ++k)
    //        {
    //            result += HexCoefficients[coefficientIndex] *
    //                phi_i[i] * phi_j[j] * phi_k[k];
    //            ++coefficientIndex;
    //        }
    //    }
    //}
    //
    intersectionPointPayload.ScalarValue = result;
    intersectionPointPayload.elementId = intersectedHexId;
    intersectionPointPayload.elementType = 0;
    intersectionPointPayload.found = 1;
    
}

#endif
