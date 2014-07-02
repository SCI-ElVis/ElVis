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

#include <ElVis/Extensions/NektarPlusPlusExtension/Expansions.cu>

__device__ __forceinline__ const ElVisFloat3& GetVertex(int hexId, int vertexId)
{
    return CoordBuffer[CoordOffsetBuffer[hexId] + vertexId];
}

__device__ __forceinline__ WorldPoint TransformReferenceToWorldHex(
    int hexId, const ReferencePoint& p)
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

    ElVisFloat x = MAKE_FLOAT(.125) * (
        t1*GetVertex(hexId, 0).x + t2*GetVertex(hexId, 1).x +
        t3*GetVertex(hexId, 2).x + t4*GetVertex(hexId, 3).x +
        t5*GetVertex(hexId, 4).x + t6*GetVertex(hexId, 5).x +
        t7*GetVertex(hexId, 6).x + t8*GetVertex(hexId, 7).x);

    ElVisFloat y = MAKE_FLOAT(.125) * (
        t1*GetVertex(hexId, 0).y + t2*GetVertex(hexId, 1).y +
        t3*GetVertex(hexId, 2).y + t4*GetVertex(hexId, 3).y +
        t5*GetVertex(hexId, 4).y + t6*GetVertex(hexId, 5).y +
        t7*GetVertex(hexId, 6).y + t8*GetVertex(hexId, 7).y);

    ElVisFloat z = MAKE_FLOAT(.125) * (
        t1*GetVertex(hexId, 0).z + t2*GetVertex(hexId, 1).z +
        t3*GetVertex(hexId, 2).z + t4*GetVertex(hexId, 3).z +
        t5*GetVertex(hexId, 4).z + t6*GetVertex(hexId, 5).z +
        t7*GetVertex(hexId, 6).z + t8*GetVertex(hexId, 7).z);

    WorldPoint result = MakeFloat3(x, y, z);
    return result;
}

__device__ __forceinline__ void CalculateJacobianHex(
    int hexId, const ElVisFloat3& p, ElVisFloat* J)
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

__device__ __forceinline__ void calculateInverseJacobianHex(
    int hexId, const ReferencePoint& p, ElVisFloat *inverse)
{
    // Calculate the Jacobian matrix.
    ElVisFloat J[9];
    CalculateJacobianHex(hexId, p, J);

    // Now take the inverse.
    ElVisFloat determinant = (-J[0]*J[4]*J[8]+J[0]*J[5]*J[7]
                              +J[3]*J[1]*J[8]-J[3]*J[2]*J[7]
                              -J[6]*J[1]*J[5]+J[6]*J[2]*J[4]);

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

__device__ __forceinline__ ReferencePoint TransformWorldToReferenceHex(
    int hexId, const WorldPoint& p)
{
    ElVisFloat tolerance = MAKE_FLOAT(1e-5);

    // So we first need an initial guess.  We can probably make this smarter,
    // but for now let's go with 0,0,0.
    ReferencePoint result = MakeFloat3(
        MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVisFloat inverse[9];

    int numIterations = 0;
    const int MAX_ITERATIONS = 100;

    do
    {
        WorldPoint f = TransformReferenceToWorldHex(hexId, result) - p;
        calculateInverseJacobianHex(hexId, result, inverse);

        ElVisFloat r_adjust = (inverse[0]*f.x + inverse[1]*f.y + inverse[2]*f.z);
        ElVisFloat s_adjust = (inverse[3]*f.x + inverse[4]*f.y + inverse[5]*f.z);
        ElVisFloat t_adjust = (inverse[6]*f.x + inverse[7]*f.y + inverse[8]*f.z);

        if( fabs(r_adjust) < tolerance &&
            fabs(s_adjust) < tolerance &&
            fabs(t_adjust) < tolerance )
        {
            ELVIS_PRINTF("[NEKTAR] CONVERGE = %d  result = %f %f %f\n", numIterations, result.x, result.y, result.z);
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

    ELVIS_PRINTF("[NEKTAR] DIDN'T CONVERGE   result = %f %f %f\n", numIterations, result.x, result.y, result.z);

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluateHexAtReferencePoint(
    ElVisFloat *coeffs, uint3 *modes, const ElVisFloat3& p)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    unsigned int cnt = 0;

    for(unsigned int k = 0; k < modes->z; ++k)
    {
        ElVisFloat value_k = ModifiedA(k, p.z);
        for(unsigned int j = 0; j < modes->y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, p.y);
            for(unsigned int i = 0; i < modes->x; ++i)
            {
                result += coeffs[cnt++] * ModifiedA(i, p.x) * value_j * value_k;
            }
        }
    }

    return result;
}

#endif
