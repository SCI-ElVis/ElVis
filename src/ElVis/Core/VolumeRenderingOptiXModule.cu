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

#ifndef ELVIS_VOLUME_RENDERING_MODULE_CU
#define ELVIS_VOLUME_RENDERING_MODULE_CU

#include <ElVis/Core/PrimaryRayGenerator.cu>
#include <ElVis/Core/VolumeRenderingPayload.cu>
#include <ElVis/Core/ConvertToColor.cu>
#include <ElVis/Core/FindElement.cu>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/IntervalPoint.cu>
#include <ElVis/Core/IntervalMatrix.cu>
#include <ElVis/Core/ElementId.h>
#include <ElVis/Core/ElementTraversal.cu>
#include <ElVis/Core/FieldEvaluator.cu>
#include <ElVis/Core/Cuda.h>

#include <ElVis/Math/TrapezoidalIntegration.hpp>
#include <ElVis/Core/TransferFunction.h>
#include <ElVis/Core/InnerIntegralFunctor.cu>
#include <ElVis/Core/GaussKronrod.cu>
#include <ElVis/Core/FieldTrapezoidalIntegration.cu>

// This file is meant to be included from a higher level .cu file which has already managed
// the many of the necessary header inclusions.


rtBuffer<ElVisFloat> OpacityBreakpoints;
rtBuffer<ElVisFloat2> OpacityTransferFunction;

rtBuffer<ElVisFloat> IntensityBreakpoints;
rtBuffer<ElVisFloat2> IntensityTransferFunction;


rtDeclareVariable(ElVisFloat, desiredH, , );

// Data members for the transfer function.
rtBuffer<ElVisFloat> DensityBreakpoints;
rtBuffer<ElVisFloat> RedBreakpoints;
rtBuffer<ElVisFloat> GreenBreakpoints;
rtBuffer<ElVisFloat> BlueBreakpoints;

rtBuffer<ElVisFloat> DensityValues;
rtBuffer<ElVisFloat> RedValues;
rtBuffer<ElVisFloat> GreenValues;
rtBuffer<ElVisFloat> BlueValues;

__device__ void GenerateTransferFunction(ElVis::TransferFunction& transferFunction)
{
    transferFunction.BlueBreakpoints() = &BlueBreakpoints[0];
    transferFunction.RedBreakpoints() = &RedBreakpoints[0];
    transferFunction.GreenBreakpoints() = &GreenBreakpoints[0];
    transferFunction.DensityBreakpoints() = &DensityBreakpoints[0];

    transferFunction.BlueValues() = &BlueValues[0];
    transferFunction.GreenValues() = &GreenValues[0];
    transferFunction.RedValues() = &RedValues[0];
    transferFunction.DensityValues() = &DensityValues[0];

    transferFunction.NumBlueBreakpoints() = BlueBreakpoints.size();
    transferFunction.NumRedBreakpoints() = RedBreakpoints.size();
    transferFunction.NumGreenBreakpoints() = GreenBreakpoints.size();
    transferFunction.NumDensityBreakpoints() = DensityBreakpoints.size();
}


template<typename F, typename FPrime>
ELVIS_DEVICE int ContainsRoot(const F& f, const FPrime& fprime, const IntervalPoint& initialGuess)
{
//    ELVIS_PRINTF("ContainsRoot with initial guess (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 initialGuess.x.GetLow(), initialGuess.x.GetHigh(),
//                 initialGuess.y.GetLow(), initialGuess.y.GetHigh(),
//                 initialGuess.z.GetLow(), initialGuess.z.GetHigh());

    IntervalPoint Xk = initialGuess;
    ElVisFloat3 yk = Xk.GetMidpoint();
//    ELVIS_PRINTF("Contains Root with yk (%2.15f, %2.15f, %2.15f\n",
//                 yk.x, yk.y, yk.z);
    ElVis::IntervalMatrix<3,3> Jk = fprime(Xk);
//    ELVIS_PRINTF("ContainsRoot with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[0].GetLow(), Jk[0].GetHigh(),
//                 Jk[1].GetLow(), Jk[1].GetHigh(),
//                 Jk[2].GetLow(), Jk[2].GetHigh());
//    ELVIS_PRINTF("ContainsRoot with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[3].GetLow(), Jk[3].GetHigh(),
//                 Jk[4].GetLow(), Jk[4].GetHigh(),
//                 Jk[5].GetLow(), Jk[5].GetHigh());
//    ELVIS_PRINTF("ContainsRoot with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[6].GetLow(), Jk[6].GetHigh(),
//                 Jk[7].GetLow(), Jk[7].GetHigh(),
//                 Jk[8].GetLow(), Jk[8].GetHigh());
    ElVis::Matrix<3,3> mid = Jk.GetMidpoint();
    ElVis::Matrix<3,3> Yk = Invert(mid);
//    ELVIS_PRINTF("ContainsRoot with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[0], Yk[1], Yk[2]);
//    ELVIS_PRINTF("ContainsRoot with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[3], Yk[4], Yk[5]);
//    ELVIS_PRINTF("ContainsRoot with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[6], Yk[7], Yk[8]);

    ElVis::IntervalMatrix<3,3> I;
    IntervalPoint Zk = Xk - yk;
//    ELVIS_PRINTF("ContainsRoot with Zk  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Zk[0].GetLow(), Zk[0].GetHigh(),
//                 Zk[1].GetLow(), Zk[1].GetHigh(),
//                 Zk[2].GetLow(), Zk[2].GetHigh());

//    ElVis::IntervalMatrix<3,3> ExpectedIdentity = Yk*Jk;
//    ELVIS_PRINTF("ContainsRoot with ExpectedIdentity  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentity[0].GetLow(), ExpectedIdentity[0].GetHigh(),
//                     ExpectedIdentity[1].GetLow(), ExpectedIdentity[1].GetHigh(),
//                     ExpectedIdentity[2].GetLow(), ExpectedIdentity[2].GetHigh());
//        ELVIS_PRINTF("ContainsRoot with ExpectedIdentity  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentity[3].GetLow(), ExpectedIdentity[3].GetHigh(),
//                     ExpectedIdentity[4].GetLow(), ExpectedIdentity[4].GetHigh(),
//                     ExpectedIdentity[5].GetLow(), ExpectedIdentity[5].GetHigh());
//        ELVIS_PRINTF("ContainsRoot with ExpectedIdentity  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentity[6].GetLow(), ExpectedIdentity[6].GetHigh(),
//                     ExpectedIdentity[7].GetLow(), ExpectedIdentity[7].GetHigh(),
//                     ExpectedIdentity[8].GetLow(), ExpectedIdentity[8].GetHigh());

//    ElVis::IntervalMatrix<3,3> ExpectedIdentityWithoutTemp = I - ExpectedIdentity;
//    ELVIS_PRINTF("ContainsRoot with ExpectedIdentityWithoutTemp  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentityWithoutTemp[0].GetLow(), ExpectedIdentityWithoutTemp[0].GetHigh(),
//                     ExpectedIdentityWithoutTemp[1].GetLow(), ExpectedIdentityWithoutTemp[1].GetHigh(),
//                     ExpectedIdentityWithoutTemp[2].GetLow(), ExpectedIdentityWithoutTemp[2].GetHigh());
//        ELVIS_PRINTF("ContainsRoot with ExpectedIdentityWithoutTemp  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentityWithoutTemp[3].GetLow(), ExpectedIdentityWithoutTemp[3].GetHigh(),
//                     ExpectedIdentityWithoutTemp[4].GetLow(), ExpectedIdentityWithoutTemp[4].GetHigh(),
//                     ExpectedIdentityWithoutTemp[5].GetLow(), ExpectedIdentityWithoutTemp[5].GetHigh());
//        ELVIS_PRINTF("ContainsRoot with ExpectedIdentityWithoutTemp  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     ExpectedIdentityWithoutTemp[6].GetLow(), ExpectedIdentityWithoutTemp[6].GetHigh(),
//                     ExpectedIdentityWithoutTemp[7].GetLow(), ExpectedIdentityWithoutTemp[7].GetHigh(),
//                     ExpectedIdentityWithoutTemp[8].GetLow(), ExpectedIdentityWithoutTemp[8].GetHigh());

//    ElVis::IntervalMatrix<3,3> ExpectedEmpty = I - Yk*Jk;
//    ELVIS_PRINTF("ContainsRoot with ExpectedEmpty  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 ExpectedEmpty[0].GetLow(), ExpectedEmpty[0].GetHigh(),
//                 ExpectedEmpty[1].GetLow(), ExpectedEmpty[1].GetHigh(),
//                 ExpectedEmpty[2].GetLow(), ExpectedEmpty[2].GetHigh());
//    ELVIS_PRINTF("ContainsRoot with ExpectedEmpty  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 ExpectedEmpty[3].GetLow(), ExpectedEmpty[3].GetHigh(),
//                 ExpectedEmpty[4].GetLow(), ExpectedEmpty[4].GetHigh(),
//                 ExpectedEmpty[5].GetLow(), ExpectedEmpty[5].GetHigh());
//    ELVIS_PRINTF("ContainsRoot with ExpectedEmpty  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 ExpectedEmpty[6].GetLow(), ExpectedEmpty[6].GetHigh(),
//                 ExpectedEmpty[7].GetLow(), ExpectedEmpty[7].GetHigh(),
//                 ExpectedEmpty[8].GetLow(), ExpectedEmpty[8].GetHigh());

//    ElVisFloat3 adjustment = Yk*f(yk);
//    ELVIS_PRINTF("Contains Root with adjustment (%2.15f, %2.15f, %2.15f\n",
//                 adjustment.x, adjustment.y, adjustment.z);

    IntervalPoint Kk = yk - Yk*f(yk) + (I - Yk*Jk)*Zk;

//    ELVIS_PRINTF("ContainsRoot Kk (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Kk.x.GetLow(), Kk.x.GetHigh(),
//                 Kk.y.GetLow(), Kk.y.GetHigh(),
//                 Kk.z.GetLow(), Kk.z.GetHigh());

    if( Intersection(Kk, Xk).GetWidth() <= 0 )
    {
        //ELVIS_PRINTF("ContainsRoot No intersection.\n");
        return 0;
    }
    else if( Subset(Kk, Xk) )
    {
        //ELVIS_PRINTF("ContainsRoot Intersection.\n");
        return 1;
    }
    else
    {
        //ELVIS_PRINTF("ContainsRoot unsure\n");
        return 2;
    }
}

template<typename F, typename FPrime>
ELVIS_DEVICE IntervalPoint MultiNewtons(const F& f, const FPrime& fprime, const IntervalPoint& initialGuess, ElVisFloat epsilon)
{
    // Do X1 = K(X0, m(X0), Y0) intersection X_k outside the loop
    // Some code duplication, but it keeps a check for i==0 out of the
    // while loop.

//    ELVIS_PRINTF("MultiNewton with initial guess (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 initialGuess.x.GetLow(), initialGuess.x.GetHigh(),
//                 initialGuess.y.GetLow(), initialGuess.y.GetHigh(),
//                 initialGuess.z.GetLow(), initialGuess.z.GetHigh());

    IntervalPoint Xk = initialGuess;
    ElVisFloat3 yk = Xk.GetMidpoint();
    ElVis::IntervalMatrix<3,3> Jk = fprime(Xk);
//    ELVIS_PRINTF("MultiNewton with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[0].GetLow(), Jk[0].GetHigh(),
//                 Jk[1].GetLow(), Jk[1].GetHigh(),
//                 Jk[2].GetLow(), Jk[2].GetHigh());
//    ELVIS_PRINTF("MultiNewton with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[3].GetLow(), Jk[3].GetHigh(),
//                 Jk[4].GetLow(), Jk[4].GetHigh(),
//                 Jk[5].GetLow(), Jk[5].GetHigh());
//    ELVIS_PRINTF("MultiNewton with J0  (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Jk[6].GetLow(), Jk[6].GetHigh(),
//                 Jk[7].GetLow(), Jk[7].GetHigh(),
//                 Jk[8].GetLow(), Jk[8].GetHigh());
    ElVis::Matrix<3,3> mid = Jk.GetMidpoint();
    ElVis::Matrix<3,3> Yk = Invert(mid);
//    ELVIS_PRINTF("MultiNewton with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[0], Yk[1], Yk[2]);
//    ELVIS_PRINTF("MultiNewton with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[3], Yk[4], Yk[5]);
//    ELVIS_PRINTF("MultiNewton with Y0  (%2.15f, %2.15f, %2.15f)\n",
//                 Yk[6], Yk[7], Yk[8]);


    ElVis::IntervalMatrix<3,3> I;


    ElVisFloat rn_1 = (I - Yk*Jk).Norm();
    IntervalPoint Zk = Xk - yk;
//    ELVIS_PRINTF("MultiNewton with r0 (%2.15f) and Z0 (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 rn_1,
//                 Zk.x.GetLow(), Zk.x.GetHigh(),
//                 Zk.y.GetLow(), Zk.y.GetHigh(),
//                 Zk.z.GetLow(), Zk.z.GetHigh());
    ElVisFloat3 fval = f(yk);
//    ELVIS_PRINTF("MultiNewton with yk (%2.15f, %2.15f, %2.15f) and f(yk) (%2.15f, %2.15f, %2.15f)\n",
//                 yk.x, yk.y, yk.z, fval.x, fval.y, fval.z);

    ElVisFloat3 Ykfyk = Yk*f(yk);
//    ELVIS_PRINTF("MultiNewton with Ykfyk0  (%2.15f, %2.15f, %2.15f)\n",
//                 Ykfyk.x, Ykfyk.y, Ykfyk.z);

    IntervalPoint Kk = yk - Yk*f(yk) + (I - Yk*Jk)*Zk;
//    ELVIS_PRINTF("MultiNewton with Ykfyk0 (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Kk.x.GetLow(), Kk.x.GetHigh(),
//                 Kk.y.GetLow(), Kk.y.GetHigh(),
//                 Kk.z.GetLow(), Kk.z.GetHigh());
    Xk = Intersection(Xk, Kk);

    ElVis::Matrix<3,3> Yk_1 = Yk;

//    ELVIS_PRINTF("MultiNewton (X0) (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                 Xk.x.GetLow(), Xk.x.GetHigh(),
//                 Xk.y.GetLow(), Xk.y.GetHigh(),
//                 Xk.z.GetLow(), Xk.z.GetHigh());


    int i = 0;
    while(Xk.GetWidth() > epsilon && i < 10)
    {
        yk = Xk.GetMidpoint();
        IntervalPoint Zk = Xk - yk;

        ElVis::IntervalMatrix<3,3> Jk = fprime(Xk);
        ElVis::Matrix<3,3> Jmidpoint = Jk.GetMidpoint();
        ElVis::Matrix<3,3> Yk = Invert(Jmidpoint);

        ElVis::IntervalMatrix<3,3> leftNormMatrix = I - Yk*Jk;
        ElVisFloat rn = leftNormMatrix.Norm();

        if( rn > rn_1 )
        {
            Yk = Yk_1;
        }

        IntervalPoint Kk = yk - Yk*f(yk) + (I - Yk*Jk)*Zk;

//        Xk_1 = Xk;
//        Yk_1 = Yk;
//        Jk_1 = Jk;
        rn_1 = rn;

        Xk = Intersection(Xk, Kk);

//        ELVIS_PRINTF("MultiNewton (X%d) (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n",
//                     i+1,
//                     Xk.x.GetLow(), Xk.x.GetHigh(),
//                     Xk.y.GetLow(), Xk.y.GetHigh(),
//                     Xk.z.GetLow(), Xk.z.GetHigh());

        ++i;
    }

    return Xk;
}

struct ClosestRootStackEntry
{
    ELVIS_DEVICE ClosestRootStackEntry() : Point() {}
    ELVIS_DEVICE ~ClosestRootStackEntry() {}
    ELVIS_DEVICE ClosestRootStackEntry(const ClosestRootStackEntry& rhs) : Point(rhs.Point) {}
    ELVIS_DEVICE ClosestRootStackEntry& operator=(const ClosestRootStackEntry& rhs)
    {
        Point = rhs.Point;
        return *this;
    }

    IntervalPoint Point;
};

template<typename F, typename FPrime>
ELVIS_DEVICE bool FindClosestRoot(const F& func, const FPrime& fprime, const IntervalPoint& initialGuess, IntervalPoint& out,
  const CurvedFaceIdx& curvedFaceIdx)
{
    ElVisFloat tolerance = FaceTolerance;

    // So we first need an initial guess.  We can probably make this smarter, but
    // for now let's go with 0,0,0.
    //ReferencePoint result = initialGuess.GetMidpoint();
    ElVisFloat2 initialReferenceCoordinates = MakeFloat2(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    getStartingReferencePointForNewtonIteration(curvedFaceIdx, initialReferenceCoordinates);

    ReferencePoint result = MakeFloat3(initialReferenceCoordinates.x, 
      initialReferenceCoordinates.y, initialGuess[2].GetMidpoint());

//   ELVIS_PRINTF("FindClosestRoot: Initial Guess (%f, %f, %f), tolerance %2.15f\n", result.x, result.y, result.z, tolerance);

    int numIterations = 0;
    ElVis::Matrix<3,3> J;
    ElVis::Matrix<3,3> inverse;
    const int MAX_ITERATIONS = 10;
    do
    {
        ELVIS_PRINTF("Starting iteration with curr guess (%f, %f, %f).\n", result.x, result.y, result.z);
        WorldPoint f = func(result);

        fprime(result, J);
        Invert(J, inverse);

//        ELVIS_PRINTF("FindClosestRoot: f (%f, %f, %f)\n", f.x, f.y, f.z);
//        ELVIS_PRINTF("J[0] (%f, %f, %f)\n", J[0], J[1], J[2]);
//        ELVIS_PRINTF("J[1] (%f, %f, %f)\n", J[3], J[4], J[5]);
//        ELVIS_PRINTF("J[2] (%f, %f, %f)\n", J[6], J[7], J[8]);
//        ELVIS_PRINTF("I[0] (%f, %f, %f)\n", inverse[0], inverse[1], inverse[2]);
//        ELVIS_PRINTF("I[1] (%f, %f, %f)\n", inverse[3], inverse[4], inverse[5]);
//        ELVIS_PRINTF("I[2] (%f, %f, %f)\n", inverse[6], inverse[7], inverse[8]);

        ElVisFloat3 step;
        step.x = (inverse[0]*f.x + inverse[1]*f.y + inverse[2]*f.z);
        step.y = (inverse[3]*f.x + inverse[4]*f.y + inverse[5]*f.z);
        step.z = (inverse[6]*f.x + inverse[7]*f.y + inverse[8]*f.z);

//        ELVIS_PRINTF("Adjust %f, %f, %f\n", r_adjust, s_adjust, t_adjust);
        adjustNewtonStepToKeepReferencePointOnFace(result, curvedFaceIdx,
          step);

        bool test = fabsf(step.x) < tolerance;
        test &= fabsf(step.y) < tolerance;
        test &= fabsf(step.z) < tolerance;
        if( test )
        {
            out[0].Set(result.x, result.x);
            out[1].Set(result.y, result.y);
            out[2].Set(result.z, result.z);
            return true;
        }

        //ReferencePoint pointAdjust = MakeFloat3(r_adjust, s_adjust, t_adjust);
        //ReferencePoint tempResult = result - pointAdjust;

        // If point adjust is so small it wont' change result then we are done.
        //if( result.x == tempResult.x && result.y == tempResult.y && result.z == tempResult.z )
        //{
        //    return result;
        //}

        //result = tempResult;
        result -= step;
        //result.x -= r_adjust;
        //result.y -= s_adjust;
        //result.z -= t_adjust;

//        if( result.x > 1 ) result.x = 1;
//        if( result.y > 1 ) result.y = 1;
//        if( result.z > 1 ) result.z = 1;
//
//        if( result.x < 0 ) result.x = 0;
//        if( result.y < 0 ) result.y = 0;
//        if( result.z < 0 ) result.z = 0;

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

    //ELVIS_PRINTF("FindClosestRoot: Exiting no root.\n");
    out[0].Set(result.x, result.x);
    out[1].Set(result.y, result.y);
    out[2].Set(result.z, result.z);
    return false;
    //return result;
}

//template<typename F, typename FPrime>
//ELVIS_DEVICE bool FindClosestRoot(const F& f, const FPrime& fprime, const IntervalPoint& initialGuess, ElVisFloat epsilon, IntervalPoint& result)
//{
//    ElVis::Stack<ClosestRootStackEntry, 64> stack;
//    stack.Push();
//    stack.Top().Point = initialGuess;

//    ELVIS_PRINTF("FindClosestRoot\n");
//    int counter = 0;
//    while( !stack.IsEmpty() && counter < 1000)
//    {
//        ++counter;
//        ELVIS_PRINTF("Iteration %d with stack size %d\n", counter, stack.Size());
//        ClosestRootStackEntry top = stack.Top();
//        int r = ContainsRoot(f, fprime, top.Point);
//        stack.Pop();

//        if( r == 1  )
//        {
//            // Contains a unique root.  Find it.
//            result = MultiNewtons(f, fprime, top.Point, MAKE_FLOAT(1e-8));
//            ELVIS_PRINTF("****************** Found root at (%2.15f, %2.15f), (%2.15f, %2.15f), (%2.15f, %2.15f)\n", result.x.GetLow(), result.x.GetHigh(),
//                         result.y.GetLow(), result.y.GetHigh(), result.z.GetLow(), result.z.GetHigh());
//            return true;
//        }
//        else if( r == 0 )
//        {
//            ELVIS_PRINTF("************************ No root.\n" );
//        }
//        else
//        {
//            ELVIS_PRINTF("***************************** Unsure\n");

//            if( stack.HasCapacity(2) )
//            {
//                // To guarantee we get the closest intersection, push the right first, so the left comes off
//                // the stack first.
//                stack.Push();
//                ClosestRootStackEntry& right = stack.Top();

//                stack.Push();
//                ClosestRootStackEntry& left = stack.Top();


//                if( top.Point.x.GetWidth() >= top.Point.y.GetWidth() &&
//                    top.Point.x.GetWidth() >= top.Point.z.GetWidth() )
//                {
//                    left.Point.z = top.Point.z;
//                    left.Point.y = top.Point.y;
//                    left.Point.x.Set(top.Point.x.GetLow(), (top.Point.x.GetLow() + top.Point.x.GetHigh())*MAKE_FLOAT(.5));

//                    right.Point.z = top.Point.z;
//                    right.Point.y = top.Point.y;
//                    right.Point.x.Set((top.Point.x.GetLow() + top.Point.x.GetHigh())*MAKE_FLOAT(.5), top.Point.x.GetHigh());
//                }
//                else if( top.Point.y.GetWidth() >= top.Point.x.GetWidth() &&
//                        top.Point.y.GetWidth() >= top.Point.z.GetWidth() )
//                {
//                    left.Point.x = top.Point.x;
//                    left.Point.z = top.Point.z;
//                    left.Point.y.Set(top.Point.y.GetLow(), (top.Point.y.GetLow() + top.Point.y.GetHigh())*MAKE_FLOAT(.5));

//                    right.Point.x = top.Point.x;
//                    right.Point.z = top.Point.z;
//                    right.Point.y.Set((top.Point.y.GetLow() + top.Point.y.GetHigh())*MAKE_FLOAT(.5), top.Point.y.GetHigh());
//                }
//                else
//                {
//                    left.Point.x = top.Point.x;
//                    left.Point.y = top.Point.y;
//                    left.Point.z.Set(top.Point.z.GetLow(), (top.Point.z.GetLow() + top.Point.z.GetHigh())*MAKE_FLOAT(.5));

//                    right.Point.x = top.Point.x;
//                    right.Point.y = top.Point.y;
//                    right.Point.z.Set((top.Point.z.GetLow() + top.Point.z.GetHigh())*MAKE_FLOAT(.5), top.Point.z.GetHigh());
//                }
//            }
//            else
//            {
//                ELVIS_PRINTF("FindClosestRoot stack not large enough.");
//                return false;
//            }
//        }
//    }
//    return false;
//}

struct EvaluateFaceFunctor
{
    ELVIS_DEVICE ElVisFloat3 operator()(const ElVisFloat3& p) const
    {
        ElVisFloat3 result;
        FaceReferencePoint facePoint;
        facePoint.x = p.x;
        facePoint.y = p.y;
        EvaluateFace(FaceId, facePoint, result);

        result.x += -Origin.x - Direction.x*p.z;
        result.y += -Origin.y - Direction.y*p.z;
        result.z += -Origin.z - Direction.z*p.z;

        return result;
    }

    GlobalFaceIdx FaceId;
    ElVisFloat3 Origin;
    ElVisFloat3 Direction;
};

struct EvaluateFaceJacobianFunctor
{
//    ELVIS_DEVICE ElVis::IntervalMatrix<3,3> operator()(const IntervalPoint& p) const
//    {
//        ElVis::IntervalMatrix<3,3> result;

//        EvaluateFaceJacobian(FaceId, p.x, p.y, result[0], result[1], result[3], result[4], result[6], result[7]);

//        result[2].Set(-Direction.x, -Direction.x);
//        result[5].Set(-Direction.y, -Direction.y);
//        result[8].Set(-Direction.z, -Direction.z);
//        return result;
//    }

    ELVIS_DEVICE void operator()(const ElVisFloat3& p, ElVis::Matrix<3,3>& result) const
    {
        FaceReferencePoint facePoint;
        facePoint.x = p.x;
        facePoint.y = p.y;

        EvaluateFaceJacobian(FaceId, facePoint, result[0], result[1], result[3], result[4], result[6], result[7]);

        result[2] = -Direction.x;
        result[5] = -Direction.y;
        result[8] = -Direction.z;
    }

    GlobalFaceIdx FaceId;
    ElVisFloat3 Origin;
    ElVisFloat3 Direction;
};

ELVIS_DEVICE void NewtonFaceIntersection(const CurvedFaceIdx& curvedFaceIdx)
{

    // Step 1 - Bound t with bounding box intersection tests.  Note that it is possible
    // to reject this face immediately if a better intersection has already been found.
    GlobalFaceIdx globalFaceIdx = ConvertToGlobalFaceIdx(curvedFaceIdx);
    ElVisFloat3 p0 = GetFaceInfo(globalFaceIdx).MinExtent;
    ElVisFloat3 p1 = GetFaceInfo(globalFaceIdx).MaxExtent;

//    ELVIS_PRINTF("NewtonFaceIntersection: Primitive %d, min (%f, %f, %f) - max (%f, %f, %f)\n",
//                 primitiveId, p0.x, p0.y, p0.z, p1.x, p1.y, p1.z);
//    ELVIS_PRINTF("FindBoxEntranceAndExit: O (%f, %f, %f), D (%f, %f, %f)\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);
    ElVisFloat tmin, tmax;
    if( !FindBoxEntranceAndExit(ray.origin, ray.direction, p0, p1, ray.tmin, ray.tmax, tmin, tmax) )
    {
//        ELVIS_PRINTF("NewtonFaceIntersection: No intersection with bounding box.\n");
        return;
    }

//    ELVIS_PRINTF("NewtonFaceIntersection: Found intersection with bounding box for face %d at %f, %f\n", primitiveId, tmin, tmax);
//    ElVisFloat3 a = MakeFloat3(ray.origin + tmin*ray.direction);
//    ElVisFloat3 b = MakeFloat3(ray.origin + tmax*ray.direction);

//    ELVIS_PRINTF("Entrance (%f, %f, %f), Exit (%f, %f, %f)\n", a.x, a.y, a.z, b.x, b.y, b.z);
//    ElVisFloat4 v0 = GetFaceVertex(primitiveId, 0);
//    ElVisFloat4 v1 = GetFaceVertex(primitiveId, 1);
//    ElVisFloat4 v2 = GetFaceVertex(primitiveId, 2);
//    ElVisFloat4 v3 = GetFaceVertex(primitiveId, 2);

//    ELVIS_PRINTF("V0 (%f, %f, %f)\n", v0.x, v0.y, v0.z);
//    ELVIS_PRINTF("V1 (%f, %f, %f)\n", v1.x, v1.y, v1.z);
//    ELVIS_PRINTF("V2 (%f, %f, %f)\n", v2.x, v2.y, v2.z);
//    ELVIS_PRINTF("V3 (%f, %f, %f)\n", v3.x, v3.y, v3.z);

    EvaluateFaceFunctor f;
    f.FaceId = globalFaceIdx;
    f.Origin = MakeFloat3(ray.origin);
    f.Direction = MakeFloat3(ray.direction);

    EvaluateFaceJacobianFunctor fprime;
    fprime.FaceId = globalFaceIdx;
    fprime.Origin = MakeFloat3(ray.origin);
    fprime.Direction = MakeFloat3(ray.direction);

    IntervalPoint initialGuess;
    initialGuess.x.Set(MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));
    initialGuess.y.Set(MAKE_FLOAT(0.0), MAKE_FLOAT(1.0));

    // Only care about the segment of the ray in front of the camer.  This can happen
    // in the element finder routine, where we cast a ray from inside an element.
    initialGuess.z.Set(fmaxf(MAKE_FLOAT(0.0), tmin), tmax);

    IntervalPoint result;
    if( FindClosestRoot(f, fprime, initialGuess, result, curvedFaceIdx) )
    {
        ElVisFloat r = result.x.GetMidpoint();
        ElVisFloat s = result.y.GetMidpoint();
        ElVisFloat t = result.z.GetMidpoint();

        bool coordIsValid = false;
        FaceReferencePoint p;
        p.x = r;
        p.y = s;

        IsValidFaceCoordinate(globalFaceIdx, p, coordIsValid);
        if( coordIsValid && initialGuess.z.Contains(t) )
        {

            ELVIS_PRINTF("Found intersection (%2.15f, %2.15f, %2.15f)\n", r, s, t);
            if( rtPotentialIntersection(t) )
            {
                intersectedFaceGlobalIdx = globalFaceIdx;
                faceIntersectionReferencePoint.x = r;
                faceIntersectionReferencePoint.y = s;
                faceIntersectionReferencePointIsValid = true;
                rtReportIntersection(0);
            }
        }
        else
        {
            ELVIS_PRINTF("Intersection found, but out of range %2.15f.\n", t);
        }
    }
    else
    {
        ELVIS_PRINTF("No root in interval\n");
    }
}


ELVIS_DEVICE bool IsCounterClockwise(const ElVisFloat3& v0, const ElVisFloat3& v1, const ElVisFloat3& v2)
{
    ElVisFloat3 e0 = v1 - v0;
    ElVisFloat3 e1 = v0 - v2;
    ElVisFloat3 n  = cross( e0, e1 );

    ElVisFloat3 e2 = v0 - MakeFloat3(ray.origin);
    ElVisFloat va  = dot( n, e2 );

//    ELVIS_PRINTF("va %2.15f\n", va);
    return (va > 0.0);

}

ELVIS_DEVICE void TriangleIntersection(GlobalFaceIdx globalFaceIdx, const ElVisFloat3& a, const ElVisFloat3& b, const ElVisFloat3& c )
{
    //ELVIS_PRINTF("TriangleIntersection (%f, %f, %f), (%f, %f, %f), (%f, %f, %f).\n", a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    ElVisFloat3 v0 = a;
    ElVisFloat3 v1 = b;
    ElVisFloat3 v2 = c;

    if( !IsCounterClockwise(a, b, c) )
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
                if(  rtPotentialIntersection( t ) )
                {
                    //ELVIS_PRINTF("TriangleIntersection: Intersection found with triangle %d at %f\n", primitiveId, t);
                    intersectedFaceGlobalIdx = globalFaceIdx;
                    faceIntersectionReferencePoint.x = MAKE_FLOAT(-2.0);
                    faceIntersectionReferencePoint.y = MAKE_FLOAT(-2.0);
                    faceIntersectionReferencePointIsValid = false;
                    rtReportIntersection(0);
                }
            }
        }
    }
}


ELVIS_DEVICE void PlanarFaceIntersectionImpl(PlanarFaceIdx planarFaceIdx)
{
    //ELVIS_PRINTF("Planar Face Intersection: Primitve %d\n", primitiveId);
    int numVertices;
    GetNumberOfVerticesForPlanarFace(planarFaceIdx, numVertices);

    GlobalFaceIdx globalFaceIdx = ConvertToGlobalFaceIdx(planarFaceIdx);
    ElVisFloat3 p0 = GetFaceInfo(globalFaceIdx).MinExtent;
    ElVisFloat3 p1 = GetFaceInfo(globalFaceIdx).MaxExtent;

    ElVisFloat tmin, tmax;
    FindBoxEntranceAndExit(ray.origin, ray.direction, p0, p1, ray.tmin, ray.tmax, tmin, tmax);

    //ELVIS_PRINTF("PlanarFaceIntersection: Found intersection with bounding box (%2.15f, %2.15f, %2.15f) - (%2.15f, %2.15f, %2.15f) for face %d at %f, %f\n",
    //             p0.x, p0.y, p0.z,
    //             p1.x, p1.y, p1.z,
    //             primitiveId, tmin, tmax);

    ElVisFloat4 v0, v1, v2;
    GetPlanarFaceVertex(planarFaceIdx, 0, v0);
    GetPlanarFaceVertex(planarFaceIdx, 1, v1);
    GetPlanarFaceVertex(planarFaceIdx, 2, v2);

    TriangleIntersection(globalFaceIdx, MakeFloat3(v0), MakeFloat3(v1), MakeFloat3(v2));

    if( numVertices == 4 )
    {
        ElVisFloat4 v3;
        GetPlanarFaceVertex(planarFaceIdx, 3, v3);
        TriangleIntersection(globalFaceIdx,  MakeFloat3(v2),  MakeFloat3(v0),  MakeFloat3(v3));
    }
}

RT_PROGRAM void PlanarFaceIntersection(int idx)
{
  PlanarFaceIdx planarFaceIdx(idx);
  if( ray.ray_type <= 1 )
  {
    if( !GetFaceEnabled(planarFaceIdx) )
    {
      ELVIS_PRINTF("PlanarFaceIntersection: Face %d is not enabled.\n", planarFaceIdx.Value);
      return;
    }
  }

  PlanarFaceIntersectionImpl(planarFaceIdx);
}

RT_PROGRAM void CurvedFaceIntersection(int idx)
{
  CurvedFaceIdx curvedFaceIdx(idx);
  if( ray.ray_type <= 1 )
  {
    if( !GetFaceEnabled(curvedFaceIdx) )
    {
      ELVIS_PRINTF("CurvedFaceIntersection: Face %d is not enabled.\n", curvedFaceIdx.Value);
      return;
    }
  }

  NewtonFaceIntersection(curvedFaceIdx);
}

RT_PROGRAM void FaceClosestHitProgram()
{
    //ELVIS_PRINTF("FaceClosestHitProgram: Intersectin %f with face %d\n", closest_t, intersectedFaceGlobalIdx);
    volumePayload.FoundIntersection = true;
    volumePayload.IntersectionT = closest_t;
    volumePayload.FaceId = intersectedFaceGlobalIdx;
    //ELVIS_PRINTF("FaceClosestHitProgram: Found %d T %f id %d\n", volumePayload.FoundIntersection,
    //    volumePayload.IntersectionT, volumePayload.FaceId);
}

struct RiemannIntegration
{
  __device__ RiemannIntegration()
  {
    accumulatedColor = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    accumulatedDensity = MAKE_FLOAT(0.0);
  }

  ElVisFloat3 accumulatedColor;
  ElVisFloat accumulatedDensity;

  __device__ bool operator()(const Segment& seg, const ElVisFloat3& origin)
  {
    ELVIS_PRINTF("RiemannIntegration\n");
    optix::size_t2 screen = color_buffer.size();

    //uint2 pixel;
    //pixel.x = launch_index.x;
    //pixel.y = launch_index.y;

    int elementId = seg.ElementId;
    int elementTypeId = seg.ElementTypeId;

    ElVisFloat a = seg.Start;
    ElVisFloat b = seg.End;

    ElVisFloat3 dir = seg.RayDirection;
    ElVisFloat d = (b-a);

    int n = Floor(d/desiredH);

    ElVisFloat h;

    if( n <= 1 )
    {
      h = b-a;
      n = 1;
    }
    else
    {
      h= d/(ElVisFloat)(n-1);
    }

    //if( traceEnabled )
    //{
    //  ELVIS_PRINTF("Total segment range: [%2.15f, %2.15f], segment Id %d\n", segmentStart[segmentIndex], segmentEnd[segmentIndex], segmentIndex);
    //  ELVIS_PRINTF("D = %2.15f, H = %2.15f, N = %d\n", d, h, n);
    //}

    //// First test for density identically 0.  This means the segment does not contribute at
    //// all to the integral and can be skipped.
    FieldEvaluator f;
    f.Origin = origin;
    f.Direction = dir;
    f.ElementId = elementId;
    f.ElementType = elementTypeId;
    f.sampleCount = 0;
    f.FieldId = FieldId;

    ElVis::TransferFunction transferFunction;
    GenerateTransferFunction(transferFunction);
    ElVisFloat s0 = f(a);
    ElVisFloat d0 = transferFunction.Sample(ElVis::eDensity, s0);
    ElVisFloat3 color0 = transferFunction.SampleColor(s0);
    ElVisFloat atten = expf(-accumulatedDensity);
    accumulatedColor += h*color0*d0*atten;

    accumulatedDensity += d0*h;

    for(int i = 1; i < n; ++i)
    {
      ElVisFloat t = a+i*h;
      ElVisFloat sample = f(t);
      ElVisFloat densityValue = transferFunction.Sample(ElVis::eDensity, sample);

      ElVisFloat3 sampleColor = transferFunction.SampleColor(sample);

      ElVisFloat atten = expf(-accumulatedDensity);

      accumulatedColor += h*sampleColor*densityValue*atten;

      accumulatedDensity += densityValue*h;
    }

    return false;
  }
};

RT_PROGRAM void PerformVolumeRendering()
{
  RiemannIntegration integrator;
  ElementTraversal(integrator);

  if( integrator.accumulatedDensity > MAKE_FLOAT(0.0) )
  {
    ElVisFloat3 finalColor = integrator.accumulatedColor +
      expf(-integrator.accumulatedDensity)*BGColor;
    raw_color_buffer[launch_index] = finalColor;
      color_buffer[launch_index] = ConvertToColor(finalColor);
  }
}

#endif
