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

#ifndef ELVIS_PRISM_CU
#define ELVIS_PRISM_CU

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

rtDeclareVariable(int, intersectedPrismId, attribute IntersectedHex, );

__device__ __forceinline__ const ElVisFloat4& GetPrismVertex(int prismId, int vertexId)
{
    return PrismVertexBuffer[prismId*6 + vertexId];
}

__device__ __forceinline__ const float4& GetPrismPlane(int prismId, int planeId)
{
    return PrismPlaneBuffer[prismId*8 + planeId];
}

__device__ __forceinline__ void GetPrismWorldToReferenceJacobian(int prismId, const ReferencePoint& p, ElVis::Matrix<3, 3>& J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = MAKE_FLOAT(1.0)-s;
    ElVisFloat t2 = t1*GetPrismVertex(prismId, 0).x;
    ElVisFloat t4 = MAKE_FLOAT(1.0)+s;
    ElVisFloat t6 = t4*GetPrismVertex(prismId, 3).x;
    ElVisFloat t8 = r+t;
    ElVisFloat t10 = MAKE_FLOAT(1.0)+r;
    ElVisFloat t14 = MAKE_FLOAT(1.0)+t;
    ElVisFloat t21 = t1*GetPrismVertex(prismId, 0).y;
    ElVisFloat t24 = t4*GetPrismVertex(prismId, 3).y;
    ElVisFloat t36 = t1*GetPrismVertex(prismId, 0).z;
    ElVisFloat t39 = t4*GetPrismVertex(prismId, 3).z;

    J[0] = -t2/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 1).x/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 2).x/MAKE_FLOAT(4.0)-t6/MAKE_FLOAT(4.0);

    J[1] = t8*GetPrismVertex(prismId, 0).x/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismId, 1).x/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismId, 2).x/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismId, 3).x/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismId, 4).x/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismId, 5).x/MAKE_FLOAT(4.0);

    J[2] = -t2/MAKE_FLOAT(4.0)-t6/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 4).x/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 5).x/MAKE_FLOAT(4.0);

    J[3] = -t21/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 1).y/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 2).y/MAKE_FLOAT(4.0)-t24/MAKE_FLOAT(4.0);

    J[4] = t8*GetPrismVertex(prismId, 0).y/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismId, 1).y/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismId, 2).y/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismId, 3).y/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismId, 4).y/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismId, 5).y/MAKE_FLOAT(4.0);

    J[5] = -t21/MAKE_FLOAT(4.0)-t24/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 4).y/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 5).y/MAKE_FLOAT(4.0);

    J[6] = -t36/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 1).z/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 2).z/MAKE_FLOAT(4.0)-t39/MAKE_FLOAT(4.0);

    J[7] = t8*GetPrismVertex(prismId, 0).z/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismId, 1).z/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismId, 2).z/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismId, 3).z/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismId, 4).z/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismId, 5).z/MAKE_FLOAT(4.0);

    J[8] = -t36/MAKE_FLOAT(4.0)-t39/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismId, 4).z/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismId, 5).z/MAKE_FLOAT(4.0);
}


__device__ __forceinline__ void CalculateTensorToWorldSpaceMappingJacobian(int prismId, const TensorPoint& p, ElVis::Matrix<3, 3>& J)
{
    ElVisFloat a = p.x;
    ElVisFloat b = p.y;
    ElVisFloat c = p.z;
    ElVisFloat t1 = MAKE_FLOAT(1.0)-c;
    ElVisFloat t2 = MAKE_FLOAT(1.0)-b;
    ElVisFloat t3 = t1*t2/MAKE_FLOAT(2.0);
    ElVisFloat t6 = t1*t2;
    ElVisFloat t9 = MAKE_FLOAT(1.0)+b;
    ElVisFloat t10 = t1*t9;
    ElVisFloat t13 = t1*t9/MAKE_FLOAT(2.0);
    ElVisFloat t17 = MAKE_FLOAT(1.0)+a;
    ElVisFloat t18 = t17*t1;
    ElVisFloat t20 = t18/MAKE_FLOAT(2.0)-MAKE_FLOAT(1.0)+c;
    ElVisFloat t29 = MAKE_FLOAT(1.0)+c;
    ElVisFloat t35 = MAKE_FLOAT(1.0)-a;
    ElVisFloat t36 = t35*t2/MAKE_FLOAT(2.0);
    ElVisFloat t39 = t17*t2;
    ElVisFloat t42 = t17*t9;
    ElVisFloat t45 = t35*t9/MAKE_FLOAT(2.0);

    ElVisFloat v1x = GetPrismVertex(prismId, 0).x;
    ElVisFloat v2x = GetPrismVertex(prismId, 1).x;
    ElVisFloat v3x = GetPrismVertex(prismId, 2).x;
    ElVisFloat v4x = GetPrismVertex(prismId, 3).x;
    ElVisFloat v5x = GetPrismVertex(prismId, 4).x;
    ElVisFloat v6x = GetPrismVertex(prismId, 5).x;

    ElVisFloat v1y = GetPrismVertex(prismId, 0).y;
    ElVisFloat v2y = GetPrismVertex(prismId, 1).y;
    ElVisFloat v3y = GetPrismVertex(prismId, 2).y;
    ElVisFloat v4y = GetPrismVertex(prismId, 3).y;
    ElVisFloat v5y = GetPrismVertex(prismId, 4).y;
    ElVisFloat v6y = GetPrismVertex(prismId, 5).y;

    ElVisFloat v1z = GetPrismVertex(prismId, 0).z;
    ElVisFloat v2z = GetPrismVertex(prismId, 1).z;
    ElVisFloat v3z = GetPrismVertex(prismId, 2).z;
    ElVisFloat v4z = GetPrismVertex(prismId, 3).z;
    ElVisFloat v5z = GetPrismVertex(prismId, 4).z;
    ElVisFloat v6z = GetPrismVertex(prismId, 5).z;

    J[0] = -t3*v1x/MAKE_FLOAT(4.0)+t6*v2x/MAKE_FLOAT(8.0)+t10*v3x/MAKE_FLOAT(8.0)-t13*v4x/MAKE_FLOAT(4.0);
    J[1] = t20*v1x/MAKE_FLOAT(4.0)-t18*v2x/MAKE_FLOAT(8.0)+t18*v3x/MAKE_FLOAT(8.0)-t20*v4x/MAKE_FLOAT(4.0)-t29*v5x/MAKE_FLOAT(4.0)+t29
    *v6x/MAKE_FLOAT(4.0);
    J[2] = -t36*v1x/MAKE_FLOAT(4.0)-t39*v2x/MAKE_FLOAT(8.0)-t42*v3x/MAKE_FLOAT(8.0)-t45*v4x/MAKE_FLOAT(4.0)+t2*v5x/MAKE_FLOAT(4.0)+t9*
    v6x/MAKE_FLOAT(4.0);
    J[3] = -t3*v1y/MAKE_FLOAT(4.0)+t6*v2y/MAKE_FLOAT(8.0)+t10*v3y/MAKE_FLOAT(8.0)-t13*v4y/MAKE_FLOAT(4.0);
    J[4] = t20*v1y/MAKE_FLOAT(4.0)-t18*v2y/MAKE_FLOAT(8.0)+t18*v3y/MAKE_FLOAT(8.0)-t20*v4y/MAKE_FLOAT(4.0)-t29*v5y/MAKE_FLOAT(4.0)+t29
    *v6y/MAKE_FLOAT(4.0);
    J[5] = -t36*v1y/MAKE_FLOAT(4.0)-t39*v2y/MAKE_FLOAT(8.0)-t42*v3y/MAKE_FLOAT(8.0)-t45*v4y/MAKE_FLOAT(4.0)+t2*v5y/MAKE_FLOAT(4.0)+t9*
    v6y/MAKE_FLOAT(4.0);
    J[6] = -t3*v1z/MAKE_FLOAT(4.0)+t6*v2z/MAKE_FLOAT(8.0)+t10*v3z/MAKE_FLOAT(8.0)-t13*v4z/MAKE_FLOAT(4.0);
    J[7] = t20*v1z/MAKE_FLOAT(4.0)-t18*v2z/MAKE_FLOAT(8.0)+t18*v3z/MAKE_FLOAT(8.0)-t20*v4z/MAKE_FLOAT(4.0)-t29*v5z/MAKE_FLOAT(4.0)+t29
    *v6z/MAKE_FLOAT(4.0);
    J[8] = -t36*v1z/MAKE_FLOAT(4.0)-t39*v2z/MAKE_FLOAT(8.0)-t42*v3z/MAKE_FLOAT(8.0)-t45*v4z/MAKE_FLOAT(4.0)+t2*v5z/MAKE_FLOAT(4.0)+t9*
    v6z/MAKE_FLOAT(4.0);

}

__device__ __forceinline__ void CalculateInversePrismJacobian(int prismId, const ReferencePoint& p, ElVis::Matrix<3, 3>& inverse)
{
    ElVis::Matrix<3, 3> J;
    GetPrismWorldToReferenceJacobian(prismId, p, J);

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

__device__ __forceinline__ TensorPoint TransformPrismReferenceToTensor(const ReferencePoint& p)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    if( t != 1 )
    {
        ElVisFloat a = MAKE_FLOAT(2.0)*(r+MAKE_FLOAT(1.0))/(MAKE_FLOAT(1.0)-t) - MAKE_FLOAT(1.0);
        ElVisFloat b = s;
        ElVisFloat c = t;

        return MakeFloat3(a,b,c);
    }
    else
    {
        // In this case we're on the collapsed edge.
        // Pick a tensor point on the corresponding
        // face.
        // So just pick a.
        return MakeFloat3(MAKE_FLOAT(0.0), s, t);
    }
}

__device__ __forceinline__ ReferencePoint TransformPrismTensorToReference(const TensorPoint& p)
{
    ElVisFloat a = p.x;
    ElVisFloat b = p.y;
    ElVisFloat c = p.z;

    ElVisFloat r = (MAKE_FLOAT(1.0)+a)/MAKE_FLOAT(2.0) * (MAKE_FLOAT(1.0)-c) - MAKE_FLOAT(1.0);
    ElVisFloat s = b;
    ElVisFloat t = c;         

    return MakeFloat3(r,s,t);
}

__device__ __forceinline__ WorldPoint TransformPrismReferenceToWorld(int prismId, const ReferencePoint& p)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = -(r+t)*(MAKE_FLOAT(1.0)-s);
    ElVisFloat t2 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)-s);
    ElVisFloat t3 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)+s);
    ElVisFloat t4 = -(r+t)*(MAKE_FLOAT(1.0)+s);
    ElVisFloat t5 = (MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)+t);
    ElVisFloat t6 = (MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)+t);

    ElVisFloat x = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismId, 0).x + t2*GetPrismVertex(prismId, 1).x +
        t3*GetPrismVertex(prismId, 2).x + t4*GetPrismVertex(prismId, 3).x +
        t5*GetPrismVertex(prismId, 4).x + t6*GetPrismVertex(prismId, 5).x);

    ElVisFloat y = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismId, 0).y + t2*GetPrismVertex(prismId, 1).y +
        t3*GetPrismVertex(prismId, 2).y + t4*GetPrismVertex(prismId, 3).y +
        t5*GetPrismVertex(prismId, 4).y + t6*GetPrismVertex(prismId, 5).y);

    ElVisFloat z = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismId, 0).z + t2*GetPrismVertex(prismId, 1).z +
        t3*GetPrismVertex(prismId, 2).z + t4*GetPrismVertex(prismId, 3).z +
        t5*GetPrismVertex(prismId, 4).z + t6*GetPrismVertex(prismId, 5).z);

    return MakeFloat3(x, y, z);
}

__device__ __forceinline__ ReferencePoint TransformPrismWorldToReference(int prismId, const WorldPoint& p)
{
    int runs = 0;

    ElVisFloat tolerance = MAKE_FLOAT(1e-5);

    ++runs;

    //if( launch_index.x == 593 && 
    //            launch_index.y == 966)
    //{
    //    rtPrintf("##### Launch (%d, %d) Id = %d.  World Point = (%e, %e, %e)\n",
    //        launch_index.x, launch_index.y, Id, 
    //        p.x, p.y, p.z);
    //}

    // So we first need an initial guess.  We can probably make this smarter, but
    // for now let's go with 0,0,0.
    ReferencePoint result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVis::Matrix<3, 3> inverse;

    int numIterations = 0;
    const int MAX_ITERATIONS = 10;
    do
    {
        WorldPoint f = TransformPrismReferenceToWorld(prismId, result) - p;
        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  f = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        f.x, f.y, f.z);
        //}

        CalculateInversePrismJacobian(prismId, result, inverse);

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("(%e, %e, %e)\n(%e, %e, %e)\n(%e, %e, %e)\n",
        //        inverse[0], inverse[1], inverse[2],
        //        inverse[3], inverse[4], inverse[5],
        //        inverse[6], inverse[7], inverse[8]
        //        );
        //}

        ElVisFloat r_adjust = (inverse[0]*f.x + inverse[1]*f.y + inverse[2]*f.z);
        ElVisFloat s_adjust = (inverse[3]*f.x + inverse[4]*f.y + inverse[5]*f.z);
        ElVisFloat t_adjust = (inverse[6]*f.x + inverse[7]*f.y + inverse[8]*f.z);

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  Adjustments = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        r_adjust, s_adjust, t_adjust);
        //}

        ////cout << "Point to transform to reference: " << p << endl;
        ////cout << "F: " << f << endl;
        ////cout << "Result: " << transformReferenceToWorld(result) << endl;

        bool test = fabsf(r_adjust) < tolerance;
        test &= fabsf(s_adjust) < tolerance;
        test &= fabsf(t_adjust) < tolerance;
        if( test ) return result;

        ReferencePoint pointAdjust = MakeFloat3(r_adjust, s_adjust, t_adjust);


        ReferencePoint tempResult = result - pointAdjust;

        // If point adjust is so small it wont' change result then we are done.
        if( result.x == tempResult.x && result.y == tempResult.y && result.z == tempResult.z )
        {
            //rtPrintf("Finished because adjustment is too small.\n");
            return result;
        }

        result = tempResult;


        //WorldPoint inversePoint = TransformPrismReferenceToWorld(prismId, result);
        ////if( launch_index.x == 593 && 
        ////            launch_index.y == 966)
        ////{
        ////    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  inversePoint = (%e, %e, %e)\n",
        ////        launch_index.x, launch_index.y, Id,
        ////        numIterations,
        ////        inversePoint.x, inversePoint.y, inversePoint.z);
        ////}

        //if( p.x == inversePoint.x &&
        //    p.y == inversePoint.y &&
        //    p.z == inversePoint.z  )
        //{
        //    //rtPrintf("Finished because transformation is exact.\n");
        //    //rtPrintf("World point: (%f, %f, %f)\n", p.x, p.y, p.z);
        //    //rtPrintf("Tensor point: (%f, %f, %f)\n", result.x, result.y, result.z);
        //    //for(int i = 0; i < 8; ++i)
        //    //{
        //    //    rtPrintf("V%d: (%f, %f, %f)\n", i, vertex_buffer[i].x, vertex_buffer[i].y, vertex_buffer[i].z);
        //    //}
        //    
        //    return result;
        //}

        ++numIterations;
    }
    while( numIterations < MAX_ITERATIONS);
    //if( numIterations == MAX_ITERATIONS )
    //{
    //    rtPrintf("Hit max iterations.\n");
    //}

    return result;
}


__device__ __forceinline__ TensorPoint TransformPrismWorldToTensor(int prismId, const WorldPoint& p)
{
    ReferencePoint ref = TransformPrismWorldToReference(prismId, p);
    TensorPoint result = TransformPrismReferenceToTensor(ref);
    return result;
}



__device__ __forceinline__ bool IntersectsPrismFace(int prismId, unsigned int faceNumber,
                               ElVisFloat4* p, const ElVisFloat3& origin, const ElVisFloat3& direction,
                               ElVisFloat& t)
{
    uint4 index = Prismvertex_face_index[faceNumber];
    bool result = false;
    if( ContainsOrigin(p[index.x], p[index.y], p[index.z], p[index.w]) )
    {
        result = FindPlaneIntersection(origin, direction, GetPrismPlane(prismId, faceNumber), t);
    }

    return result;
}

RT_PROGRAM void PrismIntersection(int prismId)
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
        M3*GetPrismVertex(prismId, 0),
        M3*GetPrismVertex(prismId, 1),
        M3*GetPrismVertex(prismId, 2),
        M3*GetPrismVertex(prismId, 3),
        M3*GetPrismVertex(prismId, 4),
        M3*GetPrismVertex(prismId, 5)};

    //if( launch_index.x == 755 && launch_index.y == 270 )
    //{
    //    rtPrintf("Testing Hex %d\n", hexId);
    //    rtPrintf("Ray Origin (%e, %e, %e)\n", origin.x, origin.y, origin.z);
    //    rtPrintf("Ray Direction (%e, %e, %e)\n", W.x, W.y, W.z);
    //
    //    for(unsigned int i = 0; i < 8; ++i)
    //    {
    //        rtPrintf("V[%d] = (%e, %e, %e)\n", i,GetPrismVertex(hexId, i).x, GetPrismVertex(hexId, i).y, GetPrismVertex(hexId, i).z);
    //    }

    //    rtPrintf("U (%e, %e, %e), V (%e, %e, %e), W (%e, %e, %e)\n",
    //        U.x, U.y, U.z, V.x, V.y, V.z, W.x, W.y, W.z);
    //    for(unsigned int i = 0; i < 8; ++i)
    //    {
    //        rtPrintf("P[%d] = (%e, %e, %e)\n", i, p[i].x, p[i].y, p[i].z);
    //    }
    //}

    for(unsigned int faceId = 0; faceId < 5; ++faceId)
    {
        ElVisFloat t = MAKE_FLOAT(-1.0);
        if( IntersectsPrismFace(prismId, faceId, p, origin, W, t) )
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
                intersectedPrismId = prismId;
                volumePayload.FoundIntersection = 1;
                volumePayload.ElementId = prismId;
                volumePayload.ElementTypeId = 1;
                volumePayload.IntersectionT = t;
                rtReportIntersection(0);
            }
        }
    }
}

RT_PROGRAM void PrismContainsOriginByCheckingPoint(int prismId)
{ 
    WorldPoint origin = MakeFloat3(ray.origin);
    // All planes point out, so each plane needs to return <= 0.
    ElVisFloat p0 = EvaluatePlane(GetPrismPlane(prismId, 0), origin);
    ElVisFloat p1 = EvaluatePlane(GetPrismPlane(prismId, 1), origin);
    ElVisFloat p2 = EvaluatePlane(GetPrismPlane(prismId, 2), origin);
    ElVisFloat p3 = EvaluatePlane(GetPrismPlane(prismId, 3), origin);
    ElVisFloat p4 = EvaluatePlane(GetPrismPlane(prismId, 4), origin);
  
    //if( launch_index.x == 593 && 
    //                launch_index.y == 966)
    //{
    //    rtPrintf("Testing Element %d with %f, %f, %f, %f, %f\n", Id, p0, p1, p2, p3, p4);
    //    //for(int i = 0; i < 6; ++i)
    //    //{
    //    //    rtPrintf("Element %d Face %d: (%f, %f, %f, %f)\n", Id, i, plane_buffer[i].x, plane_buffer[i].y, plane_buffer[i].z, plane_buffer[i].w);
    //    //}
    //    rtPrintf("Testing prism %d with start point (%f, %f, %f)\n", Id, ray.origin.x, ray.origin.y, ray.origin.z);
    //    //
    //    for(int i = 0; i < 6; ++i)
    //    {
    //        rtPrintf("Element %d  Test Point (%f, %f, %f) Vertex %d: (%f, %f, %f)\n", Id, 
    //            ray.origin.x, ray.origin.y, ray.origin.z, i, 
    //            vertex_buffer[i].x, vertex_buffer[i].y, vertex_buffer[i].z);    
    //    }
    //}

    if( p0 <= MAKE_FLOAT(0.001) && p1 <= MAKE_FLOAT(0.001) && p2 <= MAKE_FLOAT(0.001) && p3 <= MAKE_FLOAT(0.001) && p4 <= MAKE_FLOAT(0.001) )
    {
        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Element %d made it through plane test.\n", Id);
        //}

        // As a final check, make sure the tensor point transformation is in range.
        // This helps fix errors in plane comparisons.
        TensorPoint tp = TransformPrismWorldToTensor(prismId, origin);

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Element %d tensor value is (%e, %e, %e)\n", Id,
        //        tp.x, tp.y, tp.z);
        //}

        if( tp.x <= MAKE_FLOAT(-1.01) || tp.x >= MAKE_FLOAT(1.01) || 
            tp.y <= MAKE_FLOAT(-1.01) || tp.y >= MAKE_FLOAT(1.01) || 
            tp.z <= MAKE_FLOAT(-1.01) || tp.z >= MAKE_FLOAT(1.01) )
        {
            return;
        }

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("************************** Element %d made it through\n", Id);
        //}

        if(  rtPotentialIntersection( .1f ) ) 
        {
            intersectedPrismId = prismId;
            rtReportIntersection(0);
        }
    }
}


template<typename T, typename R>
__device__ __forceinline__ T EvaluatePrismScalarValueArrayVersion(unsigned int* modes, const T& x, const T& y, const T& z, const R* coefficients)
{
    T result(MAKE_FLOAT(0.0));

    T phi_i[15];
    T phi_j[15];

    ElVis::OrthoPoly::P(modes[0]-1, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(modes[1]-1, 0, 0, y, phi_j);


    unsigned int coefficientIndex = 0;
    for(unsigned int i = 0; i < modes[0]; ++i)
    {
        for(unsigned int j = 0; j < modes[1]; ++j)
        {
            for(unsigned int k = 0; k < modes[2] - i; ++k)
            {
                result += coefficients[coefficientIndex] *
                    phi_i[i] * phi_j[j] *
                    PrismPow((MAKE_FLOAT(1.0)-z), i) *
                        ElVis::OrthoPoly::P(k, 2*i+1, 0, z);

                ++coefficientIndex;
            }
        }
    }
    return result;
}

RT_PROGRAM void PrismBounding (int prismId, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    float3 v0 = ConvertToFloat3(GetPrismVertex(prismId, 0));
    float3 v1 = ConvertToFloat3(GetPrismVertex(prismId, 1));
    float3 v2 = ConvertToFloat3(GetPrismVertex(prismId, 2));
    float3 v3 = ConvertToFloat3(GetPrismVertex(prismId, 3));
    float3 v4 = ConvertToFloat3(GetPrismVertex(prismId, 4));
    float3 v5 = ConvertToFloat3(GetPrismVertex(prismId, 5));

    aabb->m_min.x = fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_min.y = fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_min.z = fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);
}





template<typename ElementType, typename PointType>
__device__ __forceinline__ ElementType EvaluatePrismFieldAtTensorPoint(unsigned int prismId, const PointType& p)
{
    uint3 degree = PrismDegrees[prismId];

    uint coefficientIndex = PrismCoefficientIndices[prismId];
    ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);
    unsigned int modes[] = {degree.x+1, degree.y+1, degree.z+1};
    ElementType result = ElVis::EvaluatePrismScalarValueArrayVersion(modes, p.x, p.y, p.z, coeffs);
    return result;
}

__device__ __forceinline__ ElVisFloat EvaluatePrismFieldAtWorldPoint(unsigned int prismId, const WorldPoint& worldPoint)
{
    TensorPoint tensorPoint = TransformPrismWorldToTensor(prismId, worldPoint);
    return EvaluatePrismFieldAtTensorPoint<ElVisFloat>(prismId, tensorPoint);
}

#endif
