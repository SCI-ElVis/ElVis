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

#ifndef ELVIS_EXTENSIONS_JACOBI_EXTENSION_PRISM_COMMON_CU
#define ELVIS_EXTENSIONS_JACOBI_EXTENSION_PRISM_COMMON_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>

ELVIS_DEVICE ElVisFloat PrismPow(const ElVisFloat& base, unsigned int power)
{
    //return powf(base, static_cast<ElVisFloat>(power));
    if( power == 0 ) return MAKE_FLOAT(1.0);
    if( power == 1 ) return base;

    ElVisFloat result = base;
    for(unsigned int i = 1; i < power; ++i)
    {
        result = result*base;
    }
    return result;
}

template<typename R>
ELVIS_DEVICE ElVis::Interval<R> PrismPow(const ElVis::Interval<R>& base, unsigned int power)
{
    if( power == 0 ) return ElVis::Interval<R>(MAKE_FLOAT(1.0));
    if( power == 1 ) return base;

    ElVis::Interval<R> result = base;
    for(unsigned int i = 1; i < power; ++i)
    {
        result = result*base;
    }
    return result;
}

ELVIS_DEVICE const ElVisFloat4& GetPrismVertex(const ElVisFloat4* prismVertexBuffer, int prismId, int vertexId)
{
    return prismVertexBuffer[prismId*6 + vertexId];
}

ELVIS_DEVICE const ElVisFloat4& GetPrismPlane(const ElVisFloat4* prismPlaneBuffer, int prismId, int planeId)
{
    return prismPlaneBuffer[prismId*8 + planeId];
}

ELVIS_DEVICE void GetPrismWorldToReferenceJacobian(const ElVisFloat4* prismVertexBuffer, int prismId, const ReferencePoint& p, ElVisFloat* J)
{
    ElVisFloat r = p.x;
    ElVisFloat s = p.y;
    ElVisFloat t = p.z;

    ElVisFloat t1 = MAKE_FLOAT(1.0)-s;
    ElVisFloat t2 = t1*GetPrismVertex(prismVertexBuffer, prismId, 0).x;
    ElVisFloat t4 = MAKE_FLOAT(1.0)+s;
    ElVisFloat t6 = t4*GetPrismVertex(prismVertexBuffer, prismId, 3).x;
    ElVisFloat t8 = r+t;
    ElVisFloat t10 = MAKE_FLOAT(1.0)+r;
    ElVisFloat t14 = MAKE_FLOAT(1.0)+t;
    ElVisFloat t21 = t1*GetPrismVertex(prismVertexBuffer, prismId, 0).y;
    ElVisFloat t24 = t4*GetPrismVertex(prismVertexBuffer, prismId, 3).y;
    ElVisFloat t36 = t1*GetPrismVertex(prismVertexBuffer, prismId, 0).z;
    ElVisFloat t39 = t4*GetPrismVertex(prismVertexBuffer, prismId, 3).z;

    J[0] = -t2/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 1).x/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 2).x/MAKE_FLOAT(4.0)-t6/MAKE_FLOAT(4.0);

    J[1] = t8*GetPrismVertex(prismVertexBuffer, prismId, 0).x/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismVertexBuffer, prismId, 1).x/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismVertexBuffer, prismId, 2).x/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismVertexBuffer, prismId, 3).x/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismVertexBuffer, prismId, 4).x/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismVertexBuffer, prismId, 5).x/MAKE_FLOAT(4.0);

    J[2] = -t2/MAKE_FLOAT(4.0)-t6/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 4).x/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 5).x/MAKE_FLOAT(4.0);

    J[3] = -t21/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 1).y/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 2).y/MAKE_FLOAT(4.0)-t24/MAKE_FLOAT(4.0);

    J[4] = t8*GetPrismVertex(prismVertexBuffer, prismId, 0).y/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismVertexBuffer, prismId, 1).y/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismVertexBuffer, prismId, 2).y/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismVertexBuffer, prismId, 3).y/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismVertexBuffer, prismId, 4).y/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismVertexBuffer, prismId, 5).y/MAKE_FLOAT(4.0);

    J[5] = -t21/MAKE_FLOAT(4.0)-t24/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 4).y/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 5).y/MAKE_FLOAT(4.0);

    J[6] = -t36/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 1).z/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 2).z/MAKE_FLOAT(4.0)-t39/MAKE_FLOAT(4.0);

    J[7] = t8*GetPrismVertex(prismVertexBuffer, prismId, 0).z/MAKE_FLOAT(4.0)-t10*GetPrismVertex(prismVertexBuffer, prismId, 1).z/MAKE_FLOAT(4.0)+t10*GetPrismVertex(prismVertexBuffer, prismId, 2).z/MAKE_FLOAT(4.0)-t8*GetPrismVertex(prismVertexBuffer, prismId, 3).z/MAKE_FLOAT(4.0)-
                t14*GetPrismVertex(prismVertexBuffer, prismId, 4).z/MAKE_FLOAT(4.0)+t14*GetPrismVertex(prismVertexBuffer, prismId, 5).z/MAKE_FLOAT(4.0);

    J[8] = -t36/MAKE_FLOAT(4.0)-t39/MAKE_FLOAT(4.0)+t1*GetPrismVertex(prismVertexBuffer, prismId, 4).z/MAKE_FLOAT(4.0)+t4*GetPrismVertex(prismVertexBuffer, prismId, 5).z/MAKE_FLOAT(4.0);
}

ELVIS_DEVICE void CalculateTensorToWorldSpaceMappingJacobian(const ElVisFloat4* prismVertexBuffer, int prismId, const TensorPoint& p, ElVisFloat* J)
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

    ElVisFloat v1x = GetPrismVertex(prismVertexBuffer, prismId, 0).x;
    ElVisFloat v2x = GetPrismVertex(prismVertexBuffer, prismId, 1).x;
    ElVisFloat v3x = GetPrismVertex(prismVertexBuffer, prismId, 2).x;
    ElVisFloat v4x = GetPrismVertex(prismVertexBuffer, prismId, 3).x;
    ElVisFloat v5x = GetPrismVertex(prismVertexBuffer, prismId, 4).x;
    ElVisFloat v6x = GetPrismVertex(prismVertexBuffer, prismId, 5).x;

    ElVisFloat v1y = GetPrismVertex(prismVertexBuffer, prismId, 0).y;
    ElVisFloat v2y = GetPrismVertex(prismVertexBuffer, prismId, 1).y;
    ElVisFloat v3y = GetPrismVertex(prismVertexBuffer, prismId, 2).y;
    ElVisFloat v4y = GetPrismVertex(prismVertexBuffer, prismId, 3).y;
    ElVisFloat v5y = GetPrismVertex(prismVertexBuffer, prismId, 4).y;
    ElVisFloat v6y = GetPrismVertex(prismVertexBuffer, prismId, 5).y;

    ElVisFloat v1z = GetPrismVertex(prismVertexBuffer, prismId, 0).z;
    ElVisFloat v2z = GetPrismVertex(prismVertexBuffer, prismId, 1).z;
    ElVisFloat v3z = GetPrismVertex(prismVertexBuffer, prismId, 2).z;
    ElVisFloat v4z = GetPrismVertex(prismVertexBuffer, prismId, 3).z;
    ElVisFloat v5z = GetPrismVertex(prismVertexBuffer, prismId, 4).z;
    ElVisFloat v6z = GetPrismVertex(prismVertexBuffer, prismId, 5).z;

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


ELVIS_DEVICE void CalculateInversePrismJacobian(const ElVisFloat4* prismVertexBuffer, int prismId, const ReferencePoint& p, ElVisFloat* inverse)
{
    ElVisFloat J[9];
    GetPrismWorldToReferenceJacobian(prismVertexBuffer, prismId, p, J);

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

ELVIS_DEVICE TensorPoint TransformPrismReferenceToTensor(const ReferencePoint& p)
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

ELVIS_DEVICE ReferencePoint TransformPrismTensorToReference(const TensorPoint& p)
{
    ElVisFloat a = p.x;
    ElVisFloat b = p.y;
    ElVisFloat c = p.z;

    ElVisFloat r = (MAKE_FLOAT(1.0)+a)/MAKE_FLOAT(2.0) * (MAKE_FLOAT(1.0)-c) - MAKE_FLOAT(1.0);
    ElVisFloat s = b;
    ElVisFloat t = c;

    return MakeFloat3(r,s,t);
}

ELVIS_DEVICE WorldPoint TransformPrismReferenceToWorld(const ElVisFloat4* prismVertexBuffer, int prismId, const ReferencePoint& p)
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

    ElVisFloat x = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismVertexBuffer, prismId, 0).x + t2*GetPrismVertex(prismVertexBuffer, prismId, 1).x +
        t3*GetPrismVertex(prismVertexBuffer, prismId, 2).x + t4*GetPrismVertex(prismVertexBuffer, prismId, 3).x +
        t5*GetPrismVertex(prismVertexBuffer, prismId, 4).x + t6*GetPrismVertex(prismVertexBuffer, prismId, 5).x);

    ElVisFloat y = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismVertexBuffer, prismId, 0).y + t2*GetPrismVertex(prismVertexBuffer, prismId, 1).y +
        t3*GetPrismVertex(prismVertexBuffer, prismId, 2).y + t4*GetPrismVertex(prismVertexBuffer, prismId, 3).y +
        t5*GetPrismVertex(prismVertexBuffer, prismId, 4).y + t6*GetPrismVertex(prismVertexBuffer, prismId, 5).y);

    ElVisFloat z = MAKE_FLOAT(.25) * (t1*GetPrismVertex(prismVertexBuffer, prismId, 0).z + t2*GetPrismVertex(prismVertexBuffer, prismId, 1).z +
        t3*GetPrismVertex(prismVertexBuffer, prismId, 2).z + t4*GetPrismVertex(prismVertexBuffer, prismId, 3).z +
        t5*GetPrismVertex(prismVertexBuffer, prismId, 4).z + t6*GetPrismVertex(prismVertexBuffer, prismId, 5).z);

    return MakeFloat3(x, y, z);
}

ELVIS_DEVICE ReferencePoint TransformPrismWorldToReference(const ElVisFloat4* prismVertexBuffer, int prismId, const WorldPoint& p)
{
    int runs = 0;

    ElVisFloat tolerance = MAKE_FLOAT(1e-5);

    ++runs;

    //if( launch_index.x == 593 &&
    //            launch_index.y == 966)
    //{
    //    ELVIS_PRINTF("##### Launch (%d, %d) Id = %d.  World Point = (%e, %e, %e)\n",
    //        launch_index.x, launch_index.y, Id,
    //        p.x, p.y, p.z);
    //}

    // So we first need an initial guess.  We can probably make this smarter, but
    // for now let's go with 0,0,0.
    ReferencePoint result = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    ElVisFloat inverse[9];

    int numIterations = 0;
    const int MAX_ITERATIONS = 10;
    do
    {
        WorldPoint f = TransformPrismReferenceToWorld(prismVertexBuffer, prismId, result) - p;
        //if( launch_index.x == 593 &&
        //            launch_index.y == 966)
        //{
        //    ELVIS_PRINTF("Launch (%d, %d) Id = %d Iter %d.  f = (%e, %e, %e)\n",
        //        launch_index.x, launch_index.y, Id,
        //        numIterations,
        //        f.x, f.y, f.z);
        //}

        CalculateInversePrismJacobian(prismVertexBuffer, prismId, result, inverse);

        //if( launch_index.x == 593 &&
        //            launch_index.y == 966)
        //{
        //    ELVIS_PRINTF("(%e, %e, %e)\n(%e, %e, %e)\n(%e, %e, %e)\n",
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
        //    ELVIS_PRINTF("Launch (%d, %d) Id = %d Iter %d.  Adjustments = (%e, %e, %e)\n",
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
            //ELVIS_PRINTF("Finished because adjustment is too small.\n");
            return result;
        }

        result = tempResult;


        //WorldPoint inversePoint = TransformPrismReferenceToWorld(prismId, result);
        ////if( launch_index.x == 593 &&
        ////            launch_index.y == 966)
        ////{
        ////    ELVIS_PRINTF("Launch (%d, %d) Id = %d Iter %d.  inversePoint = (%e, %e, %e)\n",
        ////        launch_index.x, launch_index.y, Id,
        ////        numIterations,
        ////        inversePoint.x, inversePoint.y, inversePoint.z);
        ////}

        //if( p.x == inversePoint.x &&
        //    p.y == inversePoint.y &&
        //    p.z == inversePoint.z  )
        //{
        //    //ELVIS_PRINTF("Finished because transformation is exact.\n");
        //    //ELVIS_PRINTF("World point: (%f, %f, %f)\n", p.x, p.y, p.z);
        //    //ELVIS_PRINTF("Tensor point: (%f, %f, %f)\n", result.x, result.y, result.z);
        //    //for(int i = 0; i < 8; ++i)
        //    //{
        //    //    ELVIS_PRINTF("V%d: (%f, %f, %f)\n", i, vertex_buffer[i].x, vertex_buffer[i].y, vertex_buffer[i].z);
        //    //}
        //
        //    return result;
        //}

        ++numIterations;
    }
    while( numIterations < MAX_ITERATIONS);
    //if( numIterations == MAX_ITERATIONS )
    //{
    //    ELVIS_PRINTF("Hit max iterations.\n");
    //}

    return result;
}

ELVIS_DEVICE TensorPoint TransformPrismWorldToTensor(const ElVisFloat4* prismVertexBuffer, int prismId, const WorldPoint& p)
{
    ReferencePoint ref = TransformPrismWorldToReference(prismVertexBuffer, prismId, p);
    TensorPoint result = TransformPrismReferenceToTensor(ref);
    return result;
}



template<typename T>
ELVIS_DEVICE T EvaluatePrismFieldAtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coefficients)
{
    T result(MAKE_FLOAT(0.0));

    T phi_i[8];
    T phi_j[8];
    T phi_k[8];
    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);


    unsigned int coefficientIndex = 0;
    T pp = MAKE_FLOAT(1.0);
    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        //T pp = PrismPow((MAKE_FLOAT(1.0)-z), i);
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            ElVis::OrthoPoly::P(degree.z-i, 2*i+1, 0, z, phi_k);
            for(unsigned int k = 0; k <= degree.z - i; ++k)
            {
                result += coefficients[coefficientIndex] *
                    phi_i[i] * phi_j[j] *
                    //PrismPow((MAKE_FLOAT(1.0)-z), i) *
                        pp*
                        //ElVis::OrthoPoly::P(k, 2*i+1, 0, z);
                        phi_k[k];

                ++coefficientIndex;
            }
        }

        pp = pp* (MAKE_FLOAT(1.0)-z);
    }
    return result;
}

template<typename T>
ELVIS_DEVICE T EvaluatePrismGradientDir1AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coefficients)
{
    T result(MAKE_FLOAT(0.0));

    T phi_i[15];
    T phi_j[15];

    ElVis::OrthoPoly::dP(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);


    uint coefficientIndex = 0;
    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z - i; ++k)
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

template<typename T>
ELVIS_DEVICE T EvaluatePrismGradientDir2AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coefficients)
{
    T result(MAKE_FLOAT(0.0));

    T phi_i[15];
    T phi_j[15];

    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::dP(degree.y, 0, 0, y, phi_j);


    uint coefficientIndex = 0;
    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z - i; ++k)
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

template<typename T>
ELVIS_DEVICE T EvaluatePrismGradientDir3AtTensorPoint(uint3 degree, const T& x, const T& y, const T& z, const ElVisFloat* coefficients)
{
    T result(MAKE_FLOAT(0.0));

    T phi_i[15];
    T phi_j[15];

    ElVis::OrthoPoly::P(degree.x, 0, 0, x, phi_i);
    ElVis::OrthoPoly::P(degree.y, 0, 0, y, phi_j);

    uint coefficientIndex = 0;

    for(unsigned int i = 0; i <= degree.x; ++i)
    {
        for(unsigned int j = 0; j <= degree.y; ++j)
        {
            for(unsigned int k = 0; k <= degree.z - i; ++k)
            {
                ElVisFloat d1 = PrismPow((1.0-z), static_cast<ElVisFloat>(i)) *
                    ElVis::OrthoPoly::dP(k, 2*i+1, 0, z);
                ElVisFloat d2 = MAKE_FLOAT(0.0);
                if( i > 0 )
                {
                    d2 = PrismPow((MAKE_FLOAT(1.0)-z), static_cast<ElVisFloat>(i-1)) * (-static_cast<ElVisFloat>(i)) *
                        ElVis::OrthoPoly::P(k, 2*i+1, 0, z);
                }

                result += coefficients[coefficientIndex] *
                    phi_i[i] * phi_j[j] * (d1+d2);

                ++coefficientIndex;
            }
        }
    }
    return result;
}


#endif
