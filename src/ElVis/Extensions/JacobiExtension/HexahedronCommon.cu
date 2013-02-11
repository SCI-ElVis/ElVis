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

#ifndef ELVIS_EXTENSIONS_JACOBI_EXTENSION_HEXAHEDRON_COMMON_CU
#define ELVIS_EXTENSIONS_JACOBI_EXTENSION_HEXAHEDRON_COMMON_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Extensions/JacobiExtension/Common.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>

__device__ const ElVisFloat4& GetVertex(const ElVisFloat4* hexVertexBuffer, int hexId, int vertexId)
{
    return hexVertexBuffer[hexId*8 + vertexId];
}

__device__ const ElVisFloat4& GetPlane(const ElVisFloat4* hexPlaneBuffer, int hexId, int planeId)
{
    return hexPlaneBuffer[hexId*8 + planeId];
}

__device__ __forceinline__ void TransformReferenceToWorld(const ElVisFloat4* hexVertexBuffer, int hexId, const ReferencePoint& p,
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


    ElVisFloat x = MAKE_FLOAT(.125) * (t1*GetVertex(hexVertexBuffer, hexId, 0).x + t2*GetVertex(hexVertexBuffer, hexId, 1).x +
        t3*GetVertex(hexVertexBuffer, hexId, 2).x + t4*GetVertex(hexVertexBuffer, hexId, 3).x +
        t5*GetVertex(hexVertexBuffer, hexId, 4).x + t6*GetVertex(hexVertexBuffer, hexId, 5).x +
        t7*GetVertex(hexVertexBuffer, hexId, 6).x + t8*GetVertex(hexVertexBuffer, hexId, 7).x);

    ElVisFloat y = MAKE_FLOAT(.125) * (t1*GetVertex(hexVertexBuffer, hexId, 0).y + t2*GetVertex(hexVertexBuffer, hexId, 1).y +
        t3*GetVertex(hexVertexBuffer, hexId, 2).y + t4*GetVertex(hexVertexBuffer, hexId, 3).y +
        t5*GetVertex(hexVertexBuffer, hexId, 4).y + t6*GetVertex(hexVertexBuffer, hexId, 5).y +
        t7*GetVertex(hexVertexBuffer, hexId, 6).y + t8*GetVertex(hexVertexBuffer, hexId, 7).y);

    ElVisFloat z = MAKE_FLOAT(.125) * (t1*GetVertex(hexVertexBuffer, hexId, 0).z + t2*GetVertex(hexVertexBuffer, hexId, 1).z +
        t3*GetVertex(hexVertexBuffer, hexId, 2).z + t4*GetVertex(hexVertexBuffer, hexId, 3).z +
        t5*GetVertex(hexVertexBuffer, hexId, 4).z + t6*GetVertex(hexVertexBuffer, hexId, 5).z +
        t7*GetVertex(hexVertexBuffer, hexId, 6).z + t8*GetVertex(hexVertexBuffer, hexId, 7).z);


    result.x = x;
    result.y = y;
    result.z = z;
}

__device__ void calculateTensorToWorldSpaceMappingJacobian(const ElVisFloat4* hexVertexBuffer, int hexId,
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



    ElVisFloat v1x = GetVertex(hexVertexBuffer, hexId, 0).x;
    ElVisFloat v2x = GetVertex(hexVertexBuffer, hexId, 1).x;
    ElVisFloat v3x = GetVertex(hexVertexBuffer, hexId, 2).x;
    ElVisFloat v4x = GetVertex(hexVertexBuffer, hexId, 3).x;
    ElVisFloat v5x = GetVertex(hexVertexBuffer, hexId, 4).x;
    ElVisFloat v6x = GetVertex(hexVertexBuffer, hexId, 5).x;
    ElVisFloat v7x = GetVertex(hexVertexBuffer, hexId, 6).x;
    ElVisFloat v8x = GetVertex(hexVertexBuffer, hexId, 7).x;

    ElVisFloat v1y = GetVertex(hexVertexBuffer, hexId, 0).y;
    ElVisFloat v2y = GetVertex(hexVertexBuffer, hexId, 1).y;
    ElVisFloat v3y = GetVertex(hexVertexBuffer, hexId, 2).y;
    ElVisFloat v4y = GetVertex(hexVertexBuffer, hexId, 3).y;
    ElVisFloat v5y = GetVertex(hexVertexBuffer, hexId, 4).y;
    ElVisFloat v6y = GetVertex(hexVertexBuffer, hexId, 5).y;
    ElVisFloat v7y = GetVertex(hexVertexBuffer, hexId, 6).y;
    ElVisFloat v8y = GetVertex(hexVertexBuffer, hexId, 7).y;

    ElVisFloat v1z = GetVertex(hexVertexBuffer, hexId, 0).z;
    ElVisFloat v2z = GetVertex(hexVertexBuffer, hexId, 1).z;
    ElVisFloat v3z = GetVertex(hexVertexBuffer, hexId, 2).z;
    ElVisFloat v4z = GetVertex(hexVertexBuffer, hexId, 3).z;
    ElVisFloat v5z = GetVertex(hexVertexBuffer, hexId, 4).z;
    ElVisFloat v6z = GetVertex(hexVertexBuffer, hexId, 5).z;
    ElVisFloat v7z = GetVertex(hexVertexBuffer, hexId, 6).z;
    ElVisFloat v8z = GetVertex(hexVertexBuffer, hexId, 7).z;

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

__device__ void calculateInverseJacobian(const ElVisFloat4* hexVertexBuffer, int hexId, const ReferencePoint& p,
                                                         ElVisFloat* inverse)
{
    //ELVIS_PRINTF("calculateInverseJacobian");
    ElVisFloat J[16];
    calculateTensorToWorldSpaceMappingJacobian(hexVertexBuffer, hexId, p, &J[0]);

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

__device__ __forceinline__ ReferencePoint TransformWorldToReference(const ElVisFloat4* hexVertexBuffer, int hexId, const WorldPoint& p)
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
        TransformReferenceToWorld(hexVertexBuffer, hexId, result, f);
        f.x -= p.x;
        f.y -= p.y;
        f.z -= p.z;

        calculateInverseJacobian(hexVertexBuffer, hexId, result, &inverse[0]);

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

__device__ __forceinline__ TensorPoint TransformWorldToTensor(const ElVisFloat4* hexVertexBuffer, int hexId, const WorldPoint& p)
{
    ReferencePoint ref = TransformWorldToReference(hexVertexBuffer, hexId, p);
    return ref;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexFieldAtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coeffs)
{
    T result(MAKE_FLOAT(0.0));

    T phi_k[8];
    T phi_j[8];
    T phi_i[8];

    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);
    ElVis::OrthoPoly::P(degree.z, 0, 0, z, phi_k);

    uint coefficientIndex = 0;

    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                result += coeffs[coefficientIndex]
                    * phi_i[i] * phi_j[j] * phi_k[k];
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir1AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coeffs)
{
    T result(MAKE_FLOAT(0.0));

    T phi_k[8];
    T phi_j[8];
    T phi_i[8];

    ElVis::OrthoPoly::dP(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);
    ElVis::OrthoPoly::P(degree.z, 0, 0, z, phi_k);

    uint coefficientIndex = 0;

    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                result += coeffs[coefficientIndex]
                    * phi_i[i] * phi_j[j] * phi_k[k];
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir2AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coeffs)
{
    T result(MAKE_FLOAT(0.0));

    T phi_k[8];
    T phi_j[8];
    T phi_i[8];

    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::dP(degree.y, 0, 0, y, phi_j);
    ElVis::OrthoPoly::P(degree.z, 0, 0, z, phi_k);

    uint coefficientIndex = 0;

    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                result += coeffs[coefficientIndex]
                    * phi_i[i] * phi_j[j] * phi_k[k];
                ++coefficientIndex;
            }
        }
    }

    return result;
}

template<typename T>
__device__ __forceinline__ T EvaluateHexGradientDir3AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coeffs)
{
    T result(MAKE_FLOAT(0.0));

    T phi_k[8];
    T phi_j[8];
    T phi_i[8];

    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);
    ElVis::OrthoPoly::dP(degree.z, 0, 0, z, phi_k);

    uint coefficientIndex = 0;

    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z; ++k)
            {
                result += coeffs[coefficientIndex]
                    * phi_i[i] * phi_j[j] * phi_k[k];
                ++coefficientIndex;
            }
        }
    }

    return result;
}


#endif
