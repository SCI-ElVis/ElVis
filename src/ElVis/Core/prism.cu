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

#include <ElVis/matrix.cu>
#include <optix_cuda.h>
#include <optix_math.h>
#include <optixu/optixu_aabb.h>
#include "matrix.cu"
#include "CutSurfacePayloads.cu"
#include "typedefs.cu"
#include "jacobi.cu"
#include "util.cu"
#include <ElVis/IsosurfacePrism.hpp>

// The vertices associated with this prism.
// Prism has 6 vertices.
rtBuffer<ElVisFloat4> PrismVertexBuffer;

// The vertices associated with each face.
// Faces 0-2 are quads and all four elements are used.
// Faces 3 and 4 are triangles
rtBuffer<uint4> Prismvertex_face_index;

// The planes associated with each face.
rtBuffer<float4> PrismPlaneBuffer;

// The coefficients to evaluate the scalar field.
rtBuffer<ElVisFloat> PrismCoefficients;
rtBuffer<uint> PrismCoefficientIndices;

rtBuffer<uint3> PrismDegrees;


rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(ElementFinderPayload, intersectionPointPayload, rtPayload, );
rtDeclareVariable(int, intersectedPrismId, attribute IntersectedHex, );

__device__ const ElVisFloat4& GetVertex(int prismId, int vertexId)
{
    return PrismVertexBuffer[prismId*6 + vertexId];
}

__device__ const float4& GetPlane(int prismId, int planeId)
{
    return PrismPlaneBuffer[prismId*5 + planeId];
}

__device__ void GetPrismWorldToReferenceJacobian(int prismId, const ReferencePoint& p, ElVis::Matrix<3, 3>& J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = MAKE_FLOAT(1.0)-s;
    ElVisFloat t2 = t1*GetVertex(prismId, 0).x;
    ElVisFloat t4 = MAKE_FLOAT(1.0)+s;
    ElVisFloat t6 = t4*GetVertex(prismId, 3).x;
    ElVisFloat t8 = r+t;
    ElVisFloat t10 = MAKE_FLOAT(1.0)+r;
    ElVisFloat t14 = MAKE_FLOAT(1.0)+t;
    ElVisFloat t21 = t1*GetVertex(prismId, 0).y;
    ElVisFloat t24 = t4*GetVertex(prismId, 3).y;
    ElVisFloat t36 = t1*GetVertex(prismId, 0).z;
    ElVisFloat t39 = t4*GetVertex(prismId, 3).z;

    J[0] = -t2/MAKE_FLOAT(4.0)+t1*GetVertex(prismId, 1).x/MAKE_FLOAT(4.0)+t4*GetVertex(prismId, 2).x/MAKE_FLOAT(4.0)-t6/MAKE_FLOAT(4.0);

    J[1] = t8*GetVertex(prismId, 0).x/4.0-t10*GetVertex(prismId, 1).x/4.0+t10*GetVertex(prismId, 2).x/4.0-t8*GetVertex(prismId, 3).x/4.0-
                t14*GetVertex(prismId, 4).x/4.0+t14*GetVertex(prismId, 5).x/4.0;

    J[2] = -t2/4.0-t6/4.0+t1*GetVertex(prismId, 4).x/4.0+t4*GetVertex(prismId, 5).x/4.0;

    J[3] = -t21/4.0+t1*GetVertex(prismId, 1).y/4.0+t4*GetVertex(prismId, 2).y/4.0-t24/4.0;

    J[4] = t8*GetVertex(prismId, 0).y/4.0-t10*GetVertex(prismId, 1).y/4.0+t10*GetVertex(prismId, 2).y/4.0-t8*GetVertex(prismId, 3).y/4.0-
                t14*GetVertex(prismId, 4).y/4.0+t14*GetVertex(prismId, 5).y/4.0;

    J[5] = -t21/4.0-t24/4.0+t1*GetVertex(prismId, 4).y/4.0+t4*GetVertex(prismId, 5).y/4.0;

    J[6] = -t36/4.0+t1*GetVertex(prismId, 1).z/4.0+t4*GetVertex(prismId, 2).z/4.0-t39/4.0;

    J[7] = t8*GetVertex(prismId, 0).z/4.0-t10*GetVertex(prismId, 1).z/4.0+t10*GetVertex(prismId, 2).z/4.0-t8*GetVertex(prismId, 3).z/4.0-
                t14*GetVertex(prismId, 4).z/4.0+t14*GetVertex(prismId, 5).z/4.0;

    J[8] = -t36/4.0-t39/4.0+t1*GetVertex(prismId, 4).z/4.0+t4*GetVertex(prismId, 5).z/4.0;
}


__device__ void CalculateTensorToWorldSpaceMappingJacobian(int prismId, const TensorPoint& p, ElVis::Matrix<3, 3>& J)
{
    ElVisFloat a = p.x;
    ElVisFloat b = p.y;
    ElVisFloat c = p.z;
    ElVisFloat t1 = 1.0f-c;
    ElVisFloat t2 = 1.0f-b;
    ElVisFloat t3 = t1*t2/2.0f;
    ElVisFloat t6 = t1*t2;
    ElVisFloat t9 = 1.0f+b;
    ElVisFloat t10 = t1*t9;
    ElVisFloat t13 = t1*t9/2.0;
    ElVisFloat t17 = 1.0+a;
    ElVisFloat t18 = t17*t1;
    ElVisFloat t20 = t18/2.0-1.0+c;
    ElVisFloat t29 = 1.0+c;
    ElVisFloat t35 = 1.0-a;
    ElVisFloat t36 = t35*t2/2.0;
    ElVisFloat t39 = t17*t2;
    ElVisFloat t42 = t17*t9;
    ElVisFloat t45 = t35*t9/2.0;

    ElVisFloat v1x = GetVertex(prismId, 0).x;
    ElVisFloat v2x = GetVertex(prismId, 1).x;
    ElVisFloat v3x = GetVertex(prismId, 2).x;
    ElVisFloat v4x = GetVertex(prismId, 3).x;
    ElVisFloat v5x = GetVertex(prismId, 4).x;
    ElVisFloat v6x = GetVertex(prismId, 5).x;

    ElVisFloat v1y = GetVertex(prismId, 0).y;
    ElVisFloat v2y = GetVertex(prismId, 1).y;
    ElVisFloat v3y = GetVertex(prismId, 2).y;
    ElVisFloat v4y = GetVertex(prismId, 3).y;
    ElVisFloat v5y = GetVertex(prismId, 4).y;
    ElVisFloat v6y = GetVertex(prismId, 5).y;

    ElVisFloat v1z = GetVertex(prismId, 0).z;
    ElVisFloat v2z = GetVertex(prismId, 1).z;
    ElVisFloat v3z = GetVertex(prismId, 2).z;
    ElVisFloat v4z = GetVertex(prismId, 3).z;
    ElVisFloat v5z = GetVertex(prismId, 4).z;
    ElVisFloat v6z = GetVertex(prismId, 5).z;

    J[0] = -t3*v1x/4.0+t6*v2x/8.0+t10*v3x/8.0-t13*v4x/4.0;
    J[1] = t20*v1x/4.0-t18*v2x/8.0+t18*v3x/8.0-t20*v4x/4.0-t29*v5x/4.0+t29
    *v6x/4.0;
    J[2] = -t36*v1x/4.0-t39*v2x/8.0-t42*v3x/8.0-t45*v4x/4.0+t2*v5x/4.0+t9*
    v6x/4.0;
    J[3] = -t3*v1y/4.0+t6*v2y/8.0+t10*v3y/8.0-t13*v4y/4.0;
    J[4] = t20*v1y/4.0-t18*v2y/8.0+t18*v3y/8.0-t20*v4y/4.0-t29*v5y/4.0+t29
    *v6y/4.0;
    J[5] = -t36*v1y/4.0-t39*v2y/8.0-t42*v3y/8.0-t45*v4y/4.0+t2*v5y/4.0+t9*
    v6y/4.0;
    J[6] = -t3*v1z/4.0+t6*v2z/8.0+t10*v3z/8.0-t13*v4z/4.0;
    J[7] = t20*v1z/4.0-t18*v2z/8.0+t18*v3z/8.0-t20*v4z/4.0-t29*v5z/4.0+t29
    *v6z/4.0;
    J[8] = -t36*v1z/4.0-t39*v2z/8.0-t42*v3z/8.0-t45*v4z/4.0+t2*v5z/4.0+t9*
    v6z/4.0;

}

__device__ void calculateInverseJacobian(int prismId, const ReferencePoint& p, ElVis::Matrix<3, 3>& inverse)
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

__device__ TensorPoint TransformPrismReferenceToTensor(const ReferencePoint& p)
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

__device__ ReferencePoint TransformPrismTensorToReference(const TensorPoint& p)
{
    ElVisFloat a = p.x;
    ElVisFloat b = p.y;
    ElVisFloat c = p.z;

    ElVisFloat r = (MAKE_FLOAT(1.0)+a)/MAKE_FLOAT(2.0) * (MAKE_FLOAT(1.0)-c) - MAKE_FLOAT(1.0);
    ElVisFloat s = b;
    ElVisFloat t = c;         

    return MakeFloat3(r,s,t);
}

__device__ WorldPoint TransformPrismReferenceToWorld(int prismId, const ReferencePoint& p)
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

    ElVisFloat x = MAKE_FLOAT(.25) * (t1*GetVertex(prismId, 0).x + t2*GetVertex(prismId, 1).x +
        t3*GetVertex(prismId, 2).x + t4*GetVertex(prismId, 3).x +
        t5*GetVertex(prismId, 4).x + t6*GetVertex(prismId, 5).x);

    ElVisFloat y = MAKE_FLOAT(.25) * (t1*GetVertex(prismId, 0).y + t2*GetVertex(prismId, 1).y +
        t3*GetVertex(prismId, 2).y + t4*GetVertex(prismId, 3).y +
        t5*GetVertex(prismId, 4).y + t6*GetVertex(prismId, 5).y);

    ElVisFloat z = MAKE_FLOAT(.25) * (t1*GetVertex(prismId, 0).z + t2*GetVertex(prismId, 1).z +
        t3*GetVertex(prismId, 2).z + t4*GetVertex(prismId, 3).z +
        t5*GetVertex(prismId, 4).z + t6*GetVertex(prismId, 5).z);

    return MakeFloat3(x, y, z);
}

__device__ ReferencePoint TransformWorldToReference(int prismId, const WorldPoint& p)
{
    int runs = 0;

    ElVisFloat tolerance = 1e-5;

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
    const int MAX_ITERATIONS = 100;
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

        calculateInverseJacobian(prismId, result, inverse);

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

        if( fabs(r_adjust) < tolerance &&
            fabs(s_adjust) < tolerance &&
            fabs(t_adjust) < tolerance )
        {
            //rtPrintf("Finished because transformation is within tolerance.\n");
            //rtPrintf("World point: (%f, %f, %f)\n", p.x, p.y, p.z);
            //rtPrintf("Tensor point: (%f, %f, %f)\n", result.x, result.y, result.z);
            //for(int i = 0; i < 8; ++i)
            //{
            //    rtPrintf("V%d: (%f, %f, %f)\n", i, vertex_buffer[i].x, vertex_buffer[i].y, vertex_buffer[i].z);
            //}
            return result;
        }

        ReferencePoint pointAdjust = MakeFloat3(r_adjust, s_adjust, t_adjust);

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  Point Adjust = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        pointAdjust.x, pointAdjust.y, pointAdjust.z);
        //}

        ReferencePoint tempResult = result - pointAdjust;
        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  tempResult = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        tempResult.x, tempResult.y, tempResult.z);
        //}

        // If point adjust is so small it wont' change result then we are done.
        if( result.x == tempResult.x && result.y == tempResult.y && result.z == tempResult.z )
        {
            //rtPrintf("Finished because adjustment is too small.\n");
            return result;
        }

        result = tempResult;

        ////if( result.r() < -1.0 ) result.r() = -1.0;
        ////if( result.r() > 1.0 ) result.r() = 1.0;
        ////if( result.s() < -1.0 ) result.s() -1.0;
        ////if( result.s() > 1.0 ) result.s() = 1.0;
        ////if( result.t() < -1.0 ) result.t() = -1.0;
        ////if( result.t() > 1.0 ) result.t() = 1.0;
        //// Now check the inverse through interval arithmetic.  If the point
        //// we want is in the interval then we are done.

        WorldPoint inversePoint = TransformPrismReferenceToWorld(prismId, result);
        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Launch (%d, %d) Id = %d Iter %d.  inversePoint = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        inversePoint.x, inversePoint.y, inversePoint.z);
        //}

        if( p.x == inversePoint.x &&
            p.y == inversePoint.y &&
            p.z == inversePoint.z  )
        {
            //rtPrintf("Finished because transformation is exact.\n");
            //rtPrintf("World point: (%f, %f, %f)\n", p.x, p.y, p.z);
            //rtPrintf("Tensor point: (%f, %f, %f)\n", result.x, result.y, result.z);
            //for(int i = 0; i < 8; ++i)
            //{
            //    rtPrintf("V%d: (%f, %f, %f)\n", i, vertex_buffer[i].x, vertex_buffer[i].y, vertex_buffer[i].z);
            //}
            
            return result;
        }

        ++numIterations;
    }
    while( numIterations < MAX_ITERATIONS);
    //if( numIterations == MAX_ITERATIONS )
    //{
    //    rtPrintf("Hit max iterations.\n");
    //}

    return result;
}


__device__ TensorPoint TransformWorldToTensor(int prismId, const WorldPoint& p)
{
    ReferencePoint ref = TransformWorldToReference(prismId, p);
    TensorPoint result = TransformPrismReferenceToTensor(ref);
    return result;
}

__device__ ElVisFloat EvaluatePlane(const float4& plane, const WorldPoint& p)
{
    return p.x*plane.x + p.y*plane.y + p.z*plane.z + plane.w;
}

RT_PROGRAM void PrismContainsOriginByCheckingPoint(int prismId)
{ 
    WorldPoint origin = MakeFloat3(ray.origin);
    // All planes point out, so each plane needs to return <= 0.
    ElVisFloat p0 = EvaluatePlane(GetPlane(prismId, 0), origin);
    ElVisFloat p1 = EvaluatePlane(GetPlane(prismId, 1), origin);
    ElVisFloat p2 = EvaluatePlane(GetPlane(prismId, 2), origin);
    ElVisFloat p3 = EvaluatePlane(GetPlane(prismId, 3), origin);
    ElVisFloat p4 = EvaluatePlane(GetPlane(prismId, 4), origin);
  
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

    if( p0 <= 0.001f && p1 <= 0.001f && p2 <= 0.001f && p3 <= 0.001f && p4 <= 0.001f )
    {
        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Element %d made it through plane test.\n", Id);
        //}

        // As a final check, make sure the tensor point transformation is in range.
        // This helps fix errors in plane comparisons.
        TensorPoint tp = TransformWorldToTensor(prismId, origin);

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("Element %d tensor value is (%e, %e, %e)\n", Id,
        //        tp.x, tp.y, tp.z);
        //}

        if( tp.x <= -1.01f || tp.x >= 1.01f || 
            tp.y <= -1.01f || tp.y >= 1.01f || 
            tp.z <= -1.01f || tp.z >= 1.01f )
        {
            return;
        }

        //if( launch_index.x == 593 && 
        //            launch_index.y == 966)
        //{
        //    rtPrintf("************************** Element %d made it through\n", Id);
        //}

        if(  rtPotentialIntersection( .1 ) ) 
        {
            intersectedPrismId = prismId;
            rtReportIntersection(0);
        }
    }
}


RT_PROGRAM void PrismIntersection(int prismId)
{
    ////rtPrintf("Testing Prism.\n");
    //optix::Matrix<2, 3> transformationMatrix;

    //float3* u = (float3*)(transformationMatrix.getData());
    //float3* v = (float3*)(transformationMatrix.getData() + 3);
    //GenerateUVWCoordinateSystem(ray.direction, *u, *v);
    //

    //WorldPoint o = ray.origin;
    //WorldPoint p[4];
    //uint4 vertex_index = Prismvertex_face_index[primIdx];
    //p[0] = make_float3(GetVertex(prismId, vertex_index.x));
    //p[1] = make_float3(GetVertex(prismId, vertex_index.y));
    //p[2] = make_float3(GetVertex(prismId, vertex_index.z));
    //p[3] = make_float3(GetVertex(prismId, vertex_index.w));

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
    //    WorldPoint w = ray.origin;
    //    TensorPoint tp = TransformWorldToTensor(w);
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

    //        //if( (launch_index.x == 23 || launch_index.x == 22 || launch_index.x == 21))
    //        //{
    //        //    rtPrintf("Launch (%d, %d) intersects prism element %d\n", 
    //        //        launch_index.x, launch_index.y, Id);
    //        //}

    //        rtReportIntersection(0);
    //    }
    //}
}

RT_PROGRAM void PrismBounding (int prismId, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    float3 v0 = ConvertToFloat3(GetVertex(prismId, 0));
    float3 v1 = ConvertToFloat3(GetVertex(prismId, 1));
    float3 v2 = ConvertToFloat3(GetVertex(prismId, 2));
    float3 v3 = ConvertToFloat3(GetVertex(prismId, 3));
    float3 v4 = ConvertToFloat3(GetVertex(prismId, 4));
    float3 v5 = ConvertToFloat3(GetVertex(prismId, 5));

    aabb->m_min.x = fminf(fminf(fminf(fminf(fminf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_min.y = fminf(fminf(fminf(fminf(fminf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_min.z = fminf(fminf(fminf(fminf(fminf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);

    aabb->m_max.x = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.x, v1.x), v2.x), v3.x), v4.x), v5.x);
    aabb->m_max.y = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.y, v1.y), v2.y), v3.y), v4.y), v5.y);
    aabb->m_max.z = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(v0.z, v1.z), v2.z), v3.z), v4.z), v5.z);
}


RT_PROGRAM void EvaluatePrismScalarValue()
{
    intersectionPointPayload.found = 1;
    ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;
    TensorPoint p = TransformWorldToTensor(intersectedPrismId, worldSpaceCoordinates);
    ElVisFloat result = MAKE_FLOAT(0.0);
    uint3 degree = PrismDegrees[intersectedPrismId];

    //if( launch_index.x == 593 && 
    //                launch_index.y == 966)
    //{
    //    rtPrintf("Launch (%d, %d).  (%f, %f, %f) -> (%f, %f, %f)\n",
    //        launch_index.x, launch_index.y,
    //        worldSpaceCoordinates.x, worldSpaceCoordinates.y, worldSpaceCoordinates.z,
    //        p.x, p.y, p.z);
    //}
    
    uint coefficientIndex = PrismCoefficientIndices[intersectedPrismId];
    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z - i; ++k)
            {
                result += PrismCoefficients[coefficientIndex] *
                    ElVis::OrthoPoly::P(i, 0, 0, p.x)*
                    ElVis::OrthoPoly::P(j, 0, 0, p.y)*
                    powf((1.0-p.z), static_cast<float>(i)) *
                        ElVis::OrthoPoly::P(k, 2*i+1, 0, p.z);

                //if( launch_index.x == 593 && 
                //    launch_index.y == 966)
                //{
                //    rtPrintf("Launch (%d, %d) Evaluating prism element %d and coefficient %f, term 1 = %f, term 2 = %f, term 3.1 = %f, term 3.2 = %f, and cum %f\n", 
                //        launch_index.x, launch_index.y, Id,
                //        PrismCoefficients[coefficientIndex], 
                //        Jacobi::P(i, 0, 0, p.x),
                //        Jacobi::P(j, 0, 0, p.y),
                //        powf((1.0-p.z), static_cast<float>(i)),
                //        Jacobi::P(k, 2*i+1, 0, p.z),
                //        result);
                //}
                ++coefficientIndex;
            }
        }
    }
    
    intersectionPointPayload.ScalarValue = result;
    intersectionPointPayload.elementId = intersectedPrismId;
    intersectionPointPayload.found = 1;
    intersectionPointPayload.elementType = 1;
    intersectionPointPayload.ReferenceIntersectionPoint = p;
    //if( launch_index.x == 593 && 
    //            launch_index.y == 966)
    //{
    //    rtPrintf("Launch (%d, %d) Evaluating prism element %d at (%f, %f, %f) = %f\n", 
    //        launch_index.x, launch_index.y, Id,
    //        p.x, p.y, p.z, result);
    //    rtPrintf("Launch (%d, %d) Evaluating prism element %d at world point (%f, %f, %f) = %f\n", 
    //        launch_index.x, launch_index.y, Id,
    //        worldSpaceCoordinates.x, worldSpaceCoordinates.y, worldSpaceCoordinates.z, result);
    //}


}


RT_PROGRAM void EvaluatePrismScalarValueArrayVersion()
{
    intersectionPointPayload.found = 1;
    ElVisFloat3 worldSpaceCoordinates = intersectionPointPayload.IntersectionPoint;
    TensorPoint p = TransformWorldToTensor(intersectedPrismId, worldSpaceCoordinates);
    
    uint3 degree = PrismDegrees[intersectedPrismId];

    uint coefficientIndex = PrismCoefficientIndices[intersectedPrismId];
    ElVisFloat* coeffs = &(PrismCoefficients[coefficientIndex]);
    unsigned int modes[] = {degree.x+1, degree.y+1, degree.z+1};
    ElVisFloat result = ElVis::EvaluatePrismScalarValueArrayVersion(modes, p.x, p.y, p.z, coeffs);

    //ElVisFloat result = MAKE_FLOAT(0.0);
    //ElVisFloat phi_i[15];
    //ElVisFloat phi_j[15];

    //ElVis::OrthoPoly::P(degree.x, 0, 0, p.x, phi_i);
    //ElVis::OrthoPoly::P(degree.y, 0, 0, p.y, phi_j);

   
    //
    //for(unsigned int i = 0; i <= degree.x; ++i)
    //{
    //    for(unsigned int j = 0; j <= degree.y; ++j)
    //    {
    //        for(unsigned int k = 0; k <= degree.z - i; ++k)
    //        {
    //            result += PrismCoefficients[coefficientIndex] *
    //                phi_i[i] * phi_j[j] *
    //                powf((1.0-p.z), static_cast<float>(i)) *
    //                    ElVis::OrthoPoly::P(k, 2*i+1, 0, p.z);

    //            ++coefficientIndex;
    //        }
    //    }
    //}
    
    intersectionPointPayload.ScalarValue = result;
    intersectionPointPayload.elementId = intersectedPrismId;
    intersectionPointPayload.found = 1;
    intersectionPointPayload.elementType = 1;
    intersectionPointPayload.ReferenceIntersectionPoint = p;

}

#endif