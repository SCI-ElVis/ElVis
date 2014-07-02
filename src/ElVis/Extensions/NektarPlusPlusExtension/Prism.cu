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

#ifndef ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_PRISM_CU
#define ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_PRISM_CU

#include <ElVis/Extensions/NektarPlusPlusExtension/Expansions.cu>

__device__ __forceinline__ ElVisFloat EvaluatePrismAtReferencePoint(
    ElVisFloat *coeffs, uint3 *modes, const ElVisFloat3& p)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    unsigned int cnt = 0;

    for (uint i = 0; i < modes->x; ++i)
    {
        ElVisFloat value_i = ModifiedA(i, p.x);
        for (uint j = 0; j < modes->y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, p.y) * value_i;
            for (uint k = 0; k < modes->z - i; ++k)
            {
                result += coeffs[cnt++] * ModifiedB(i, k, p.z) * value_j;
            }
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluatePrismGradXAtReferencePoint(
    ElVisFloat *coeffs, uint3 *modes, const ElVisFloat3& p)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    unsigned int cnt = 0;

    for (uint i = 0; i < modes->x; ++i)
    {
        ElVisFloat value_i = ModifiedAPrime(i, p.x);
        for (uint j = 0; j < modes->y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, p.y);
            for (uint k = 0; k < modes->z - i; ++k)
            {
                result += coeffs[cnt++] * ModifiedB(i, k, p.z) * value_i * value_j;
            }
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluatePrismGradYAtReferencePoint(
    ElVisFloat *coeffs, uint3 *modes, const ElVisFloat3& p)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    unsigned int cnt = 0;

    for (uint i = 0; i < modes->x; ++i)
    {
        ElVisFloat value_i = ModifiedA(i, p.x);
        for (uint j = 0; j < modes->y; ++j)
        {
            ElVisFloat value_j = ModifiedAPrime(j, p.y);
            for (uint k = 0; k < modes->z - i; ++k)
            {
                result += coeffs[cnt++] * ModifiedB(i, k, p.z) * value_i * value_j;
            }
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluatePrismGradZAtReferencePoint(
    ElVisFloat *coeffs, uint3 *modes, const ElVisFloat3& p)
{
    ElVisFloat result = MAKE_FLOAT(0.0);
    unsigned int cnt = 0;

    for (uint i = 0; i < modes->x; ++i)
    {
        ElVisFloat value_i = ModifiedA(i, p.x);
        for (uint j = 0; j < modes->y; ++j)
        {
            ElVisFloat value_j = ModifiedA(j, p.y);
            for (uint k = 0; k < modes->z - i; ++k)
            {
                result += coeffs[cnt++] * ModifiedBPrime(i, k, p.z) * value_i * value_j;
            }
        }
    }

    return result;
}

__device__ __forceinline__ WorldPoint TransformReferenceToWorldLinearPrism(
    int priId, const ReferencePoint& p)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t2 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t3 = (MAKE_FLOAT(1.0)+r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t4 = (MAKE_FLOAT(1.0)-r)*(MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)-t);
    ElVisFloat t5 = (MAKE_FLOAT(1.0)-s)*(MAKE_FLOAT(1.0)+t);
    ElVisFloat t6 = (MAKE_FLOAT(1.0)+s)*(MAKE_FLOAT(1.0)+t);

    ElVisFloat x = MAKE_FLOAT(.125) * (
        t1*GetVertex(priId, 0).x + t2*GetVertex(priId, 1).x +
        t3*GetVertex(priId, 2).x + t4*GetVertex(priId, 3).x) +
            MAKE_FLOAT(.5) * (
        t5*GetVertex(priId, 4).x + t6*GetVertex(priId, 5).x);

    ElVisFloat y = MAKE_FLOAT(.125) * (
        t1*GetVertex(priId, 0).y + t2*GetVertex(priId, 1).y +
        t3*GetVertex(priId, 2).y + t4*GetVertex(priId, 3).y) +
            MAKE_FLOAT(.5) * (
        t5*GetVertex(priId, 4).y + t6*GetVertex(priId, 5).y);

    ElVisFloat z = MAKE_FLOAT(.125) * (
        t1*GetVertex(priId, 0).z + t2*GetVertex(priId, 1).z +
        t3*GetVertex(priId, 2).z + t4*GetVertex(priId, 3).z) +
            MAKE_FLOAT(.25) * (
        t5*GetVertex(priId, 4).z + t6*GetVertex(priId, 5).z);

    WorldPoint result = MakeFloat3(x, y, z);
    return result;
}

__device__ __forceinline__ WorldPoint TransformReferenceToWorldPrism(
    int priId, const ReferencePoint& p)
{
    const int offset = CurvedGeomOffsetBuffer[priId];
    uint3 *modes = &CurvedGeomNumModesBuffer[priId];

    if (offset >= 0)
    {
        const int nm = modes->x * modes->y * modes->z;
        return MakeFloat3(
            EvaluatePrismAtReferencePoint(
                &CurvedGeomBuffer[offset], modes, p),
            EvaluatePrismAtReferencePoint(
                &CurvedGeomBuffer[offset+nm], modes, p),
            EvaluatePrismAtReferencePoint(
                &CurvedGeomBuffer[offset+2*nm], modes, p));
    }
    else
    {
        return TransformReferenceToWorldLinearPrism(priId, p);
    }
}

__device__ __forceinline__ void CalculateJacobianLinearPrism(
    int priId, const ElVisFloat3& p, ElVisFloat* J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    // ElVisFloat t1 = 1.0f-s;
    // ElVisFloat t2 = 1.0f-t;
    // ElVisFloat t3 = t1*t2;
    // ElVisFloat t6 = 1.0f+s;
    // ElVisFloat t7 = t6*t2;
    // ElVisFloat t10 = 1.0f+t;
    // ElVisFloat t11 = t1*t10;
    // ElVisFloat t14 = t6*t10;
    // ElVisFloat t18 = 1.0f-r;
    // ElVisFloat t19 = t18*t2;
    // ElVisFloat t21 = 1.0f+r;
    // ElVisFloat t22 = t21*t2;
    // ElVisFloat t26 = t18*t10;
    // ElVisFloat t28 = t21*t10;
    // ElVisFloat t33 = t18*t1;
    // ElVisFloat t35 = t21*t1;
    // ElVisFloat t37 = t21*t6;
    // ElVisFloat t39 = t18*t6;

    ElVisFloat eta00 = MAKE_FLOAT(1.0) - r;
    ElVisFloat eta01 = MAKE_FLOAT(1.0) + r;
    ElVisFloat eta10 = MAKE_FLOAT(1.0) - s;
    ElVisFloat eta11 = MAKE_FLOAT(1.0) + s;
    ElVisFloat eta20 = MAKE_FLOAT(1.0) - t;
    ElVisFloat eta21 = MAKE_FLOAT(1.0) + t;

    ElVisFloat v0x = GetVertex(priId, 0).x;
    ElVisFloat v1x = GetVertex(priId, 1).x;
    ElVisFloat v2x = GetVertex(priId, 2).x;
    ElVisFloat v3x = GetVertex(priId, 3).x;
    ElVisFloat v4x = GetVertex(priId, 4).x;
    ElVisFloat v5x = GetVertex(priId, 5).x;

    ElVisFloat v0y = GetVertex(priId, 0).y;
    ElVisFloat v1y = GetVertex(priId, 1).y;
    ElVisFloat v2y = GetVertex(priId, 2).y;
    ElVisFloat v3y = GetVertex(priId, 3).y;
    ElVisFloat v4y = GetVertex(priId, 4).y;
    ElVisFloat v5y = GetVertex(priId, 5).y;

    ElVisFloat v0z = GetVertex(priId, 0).z;
    ElVisFloat v1z = GetVertex(priId, 1).z;
    ElVisFloat v2z = GetVertex(priId, 2).z;
    ElVisFloat v3z = GetVertex(priId, 3).z;
    ElVisFloat v4z = GetVertex(priId, 4).z;
    ElVisFloat v5z = GetVertex(priId, 5).z;

    J[0] = (- eta10*v0x + eta10*v1x + eta11*v2x - eta11*v3x) * eta20;
    J[1] = (- eta00*v0x - eta01*v1x + eta01*v2x + eta00*v3x) * eta20 +
        MAKE_FLOAT(2.0) * eta21 * (-v4x + v5x);
    J[2] = - eta00*eta10*v0x - eta01*eta10*v1x - eta01*eta11*v2x
        - eta00*eta11*v3x + MAKE_FLOAT(2.0) * (eta10 * v4x + eta11 * v5x);
    J[3] = (- eta10*v0y + eta10*v1y + eta11*v2y - eta11*v3y) * eta20;
    J[4] = (- eta00*v0y - eta01*v1y + eta01*v2y + eta00*v3y) * eta20 +
        MAKE_FLOAT(2.0) * eta21 * (-v4y + v5y);
    J[5] = - eta00*eta10*v0y - eta01*eta10*v1y - eta01*eta11*v2y
        - eta00*eta11*v3y + MAKE_FLOAT(2.0) * (eta10 * v4y + eta11 * v5y);
    J[6] = (- eta10*v0z + eta10*v1z + eta11*v2z - eta11*v3z) * eta20;
    J[7] = (- eta00*v0z - eta01*v1z + eta01*v2z + eta00*v3z) * eta20 +
        MAKE_FLOAT(2.0) * eta21 * (-v4z + v5z);
    J[8] = - eta00*eta10*v0z - eta01*eta10*v1z - eta01*eta11*v2z
        - eta00*eta11*v3z + MAKE_FLOAT(2.0) * (eta10 * v4z + eta11 * v5z);

    J[0] *= MAKE_FLOAT(0.125);
    J[1] *= MAKE_FLOAT(0.125);
    J[2] *= MAKE_FLOAT(0.125);
    J[3] *= MAKE_FLOAT(0.125);
    J[4] *= MAKE_FLOAT(0.125);
    J[5] *= MAKE_FLOAT(0.125);
    J[6] *= MAKE_FLOAT(0.125);
    J[7] *= MAKE_FLOAT(0.125);
    J[8] *= MAKE_FLOAT(0.125);
}

__device__ __forceinline__ void CalculateJacobianPrism(
    int priId, const ElVisFloat3& p, ElVisFloat* J)
{
    const int offset = CurvedGeomOffsetBuffer[priId];
    uint3 *modes = &CurvedGeomNumModesBuffer[priId];

    if (offset >= 0)
    {
        const int nm = modes->x * modes->y * modes->z;
        J[0] = EvaluatePrismGradXAtReferencePoint(
            &CurvedGeomBuffer[offset], modes, p);
        J[1] = EvaluatePrismGradYAtReferencePoint(
            &CurvedGeomBuffer[offset], modes, p);
        J[2] = EvaluatePrismGradZAtReferencePoint(
            &CurvedGeomBuffer[offset], modes, p);
        J[3] = EvaluatePrismGradXAtReferencePoint(
            &CurvedGeomBuffer[offset+nm], modes, p);
        J[4] = EvaluatePrismGradYAtReferencePoint(
            &CurvedGeomBuffer[offset+nm], modes, p);
        J[5] = EvaluatePrismGradZAtReferencePoint(
            &CurvedGeomBuffer[offset+nm], modes, p);
        J[6] = EvaluatePrismGradXAtReferencePoint(
            &CurvedGeomBuffer[offset+2*nm], modes, p);
        J[7] = EvaluatePrismGradYAtReferencePoint(
            &CurvedGeomBuffer[offset+2*nm], modes, p);
        J[8] = EvaluatePrismGradZAtReferencePoint(
            &CurvedGeomBuffer[offset+2*nm], modes, p);
    }
    else
    {
        CalculateJacobianLinearPrism(priId, p, J);
    }
}

__device__ __forceinline__ void calculateInverseJacobianPrism(
    int priId, const ReferencePoint& p, ElVisFloat *inverse)
{
    // Calculate the Jacobian matrix.
    ElVisFloat J[9];
    CalculateJacobianPrism(priId, p, J);

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

__device__ __forceinline__ ReferencePoint TransformWorldToReferencePrism(
    int priId, const WorldPoint& p)
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
        WorldPoint f = TransformReferenceToWorldPrism(priId, result) - p;
        calculateInverseJacobianPrism(priId, result, inverse);

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

#endif
