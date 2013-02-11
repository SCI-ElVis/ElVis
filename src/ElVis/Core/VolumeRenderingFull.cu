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

#ifndef ELVIS_VOLUME_RENDERING_FULL_CU
#define ELVIS_VOLUME_RENDERING_FULL_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/FieldEvaluator.cu>
#include <ElVis/Math/TrapezoidalIntegration.hpp>
#include <ElVis/Core/TransferFunction.h>
#include <math_functions.h>
#include <ElVis/Core/GaussKronrod.cu>

namespace ElVis
{
    enum IntegrationStatus
    {
        eFinished,
        ePartial
    };


    __device__ void ActualH(ElVisFloat a, ElVisFloat b, ElVisFloat desiredH, ElVisFloat& h, int& n)
    {
        ElVisFloat d = (b-a);
        n = Floor(d/desiredH);

        if( n == 0 )
        {
            h = d;
            n = 1;
        }
        else
        {
            h= d/(ElVisFloat)(n);
        }
    }

    struct FieldTrapezoidalIntegration
    {

        // n is the number of intervals.
        // So evaluation is at n+1 points.
        template<typename T, typename IntegrandType, typename FieldEvaluatorType>
        ELVIS_DEVICE static T CompositeIntegration(const IntegrandType& integrand, const FieldEvaluatorType& field, const T& a, const T& b, unsigned int n, bool traceEnabled)
        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Running trapezoidal rule on interval [%2.15f, %2.15f] with %d samples\n", a, b, n);
//            }

            T h = (b-a);
            if( n > 0 )
            {
                h = (b-a)/(n);
            }
            T result = 0.0;
            for(unsigned int i = 1; i < n; ++i)
            {
                ElVisFloat t = a+i*h;
                ElVisFloat s = field(t);
                ElVisFloat integrandValue = integrand(t,s);
                result += integrandValue;

//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("Trapezoidal sample at t %2.15f, field %2.15f, integrand value %2.15f\n", t, s, integrandValue);
//                }
            }

            ElVisFloat sa = field(a);
            ElVisFloat sb = field(b);

            result += .5*integrand(a, sa) + .5*integrand(b, sb);
            result*=h;

//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Finalizing Trapezoidal sample at t (%2.15f, %2.15f), field (%2.15f, %2.15f), result %2.15f and h %2.15f\n", a, b, sa, sb, result, h);
//                ELVIS_PRINTF("Finished trapezoidal on interval [%2.15f, %2.15f] with endpoint transfer samples %2.15f, %2.15f with result %2.15f\n", a, b, integrand(a, sa), integrand(b, sb), result);
//            }
            return result;
        }
    };

    struct InnerIntegralFunctor
    {
        ELVIS_DEVICE ElVisFloat GetMaxValue(const Interval<ElVisFloat>& domain) const
        {
            return transferFunction->GetMaxValue(eDensity, domain);
        }


        ELVIS_DEVICE ElVisFloat operator()(const ElVisFloat& t, const ElVisFloat& s, bool traceEnabled=false) const
        {
            return transferFunction->Sample(eDensity, s);
        }
        TransferFunction* transferFunction;
    };

    struct OuterIntegralIntegrandWithInnerTrapezoidal
    {
//            __device__ ElVisFloat GetMaxValue(const Interval<ElVisFloat>& domain) const
//            {
//                return transferFunction->GetMaxValue(channel, domain) *
//                        transferFunction->GetMaxValue(eDensity, domain);
//            }

        ELVIS_DEVICE ElVisFloat3 operator() (const ElVisFloat& t, const ElVisFloat& s, bool traceEnabled = false) const
        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Starting outer integrand at %2.15f, accumulatedDensity = %2.15f, innerT = %2.15f, innerH = %2.15f\n", t, accumulatedDensity, innerT, innerH);
//            }

            ElVisFloat3 c = transferFunction->SampleColor(s);
            ElVisFloat d = transferFunction->Sample(eDensity, s);

            int numberAdditionalInnerSamples = 0;
            ElVisFloat newH;

            ActualH(innerT, t, innerH, newH, numberAdditionalInnerSamples);

            accumulatedDensity += FieldTrapezoidalIntegration::CompositeIntegration(*innerIntegral, *field, innerT, t, numberAdditionalInnerSamples, traceEnabled);

            // If 0, then the endponits have already been calculated.
            // By setting n to a different but hopefully close h, we don't need this fixup.
//            if( numberAdditionalInnerSamples != 0 )
//            {
//                ElVisFloat t0 = t-numberAdditionalInnerSamples*innerH;
//                ElVisFloat s0 = (*field)(t0);
//                accumulatedDensity += MAKE_FLOAT(.5)*(t-t0)*( (*innerIntegral)(t0, s0) + (*innerIntegral)(t, s));
//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("Inner Integral final adjustment t0 = %2.15f, s0 = %2.15f\n", t0, s0);
//                    ELVIS_PRINTF("Sampling outer integrand at %2.15f, with color sample %2.15f, density %2.15f, and accumulated density %2.15f\n", t, c, d, accumulatedDensity);
//                }
//            }

            innerT = t;

            return c*d*exp(-accumulatedDensity);
        }

        TransferFunction* transferFunction;

        InnerIntegralFunctor* innerIntegral;
        FieldEvaluator* field;
        mutable ElVisFloat accumulatedDensity;
        mutable ElVisFloat innerT;
        mutable ElVisFloat innerH;
    };

//    struct OuterIntegralIntegrand
//    {
//        __device__ ElVisFloat GetMaxValue(const Interval<ElVisFloat>& domain) const
//        {
//            return transferFunction->GetMaxValue(channel, domain) *
//                    transferFunction->GetMaxValue(eDensity, domain);
//        }

//        __device__ ElVisFloat operator() (const ElVisFloat& t, const ElVisFloat& s) const
//        {
//            ElVisFloat c = transferFunction->Sample(channel, s);
//            ElVisFloat d = transferFunction->Sample(eDensity, s);
//            ElVisFloat accumulatedDensity = innerIntegralApproximation->SampleInnerIntegral(t, s, eDensity, transferFunction);

//            return c*d*Exp(-accumulatedDensity);
//        }

//        TransferFunction* transferFunction;
//        TransferFunctionChannel channel;
//        ReentrantAdaptiveTrapezoidal<ElVisFloat, 21>* innerIntegralApproximation;
//    };

    /// \brief epsilon - the desired global error.
    extern "C" __global__ void IntegrateSegmentSingleThreadPerRayFullVersion(ElVisFloat3 origin, const int* __restrict__ segmentElementId, const int* __restrict__ segmentElementType,
                                                                             const ElVisFloat3* __restrict__ segmentDirection,
                                                                             const ElVisFloat* __restrict__ segmentStart, const ElVisFloat* __restrict__ segmentEnd,
                                                                             int fieldId,
                                                                             TransferFunction* transferFunction, ElVisFloat epsilon, bool enableTrace,
                                                                             ElVisFloat* __restrict__ densityAccumulator, ElVisFloat3* __restrict__ colorAccumulator)
    {
//        int2 trace = make_int2(512/2, 512/2);

//        // Step 1 - Categorize the field along the segment.
//        //
//        // Case 1: Total space skipping.  When the density function is identically 0 over the entire segment, then there is nothing to do.
//        // Case 2: Density only.  If the color components are identially 0, then we only need to integrate the density contribution.
//        //    2.1: No breakpoints.  Use gauss-kronrod on the entire interval.  The 7-15 rule is probably sufficient.
//        //    2.2: Breakpoints.  Adaptive trapezoidal.
//        // Case 3: Everything.  Both density and color contribute.
//        //    2.1.  No breakpoints in either.
//        //    2.2.  Color breakpoints, density no
//        //    2.3.  Color no breakpoints, density yes.
//        //    3.5  Color and density have breakpoints.
//        // First pass - do adaptive trapezoidal on the density because I know I can then evaluate at any point within my error
//        // budget.  Then I can do adaptive or gauss-kronrod on the main integral without a problem.

//        // If the color is doing gauss-kronrod, then I can incrementally evaluate the density using adaptive trapezoidal and don't need to keep the entire
//        // structure around.  It is only when the outer integral is adaptive that I need to do that (case 2.2).

//        // In cases 2 and 3, each component can be integrated differently based on breakpoints.

//        // Density integration

//        uint2 pixel;
//        pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
//        pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

//        bool traceEnabled = (pixel.x == trace.x && pixel.y == trace.y && enableTrace);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Esilon %2.10f\n", epsilon);
//        }
//        uint2 screen;
//        screen.x = gridDim.x * blockDim.x;
//        screen.y = gridDim.y * blockDim.y;

//        int segmentIndex = pixel.x + screen.x*pixel.y;
//        if( segmentEnd[segmentIndex] < MAKE_FLOAT(0.0) )
//        {
//            return;
//        }

//        int elementId = segmentElementId[segmentIndex];
//        if( elementId == -1 )
//        {
//            return;
//        }

//        int elementTypeId = segmentElementType[segmentIndex];
//        ElVisFloat accumulatedDensity = densityAccumulator[segmentIndex];
//        ElVisFloat3 color = colorAccumulator[segmentIndex];
//        ElVisFloat a = segmentStart[segmentIndex];
//        ElVisFloat b = segmentEnd[segmentIndex];
////        if( traceEnabled )
////        {
////            b = 2.024846;
////        }
//        ElVisFloat3 dir = segmentDirection[segmentIndex];
//        ElVisFloat d = (b-a);

//        if( d == MAKE_FLOAT(0.0) )
//        {
//            return;
//        }


//        // First test for density identically 0.  This means the segment does not contribute at
//        // all to the integral and can be skipped.
//        ElVisFloat3 p0 = origin + a*dir;
//        ElVisFloat3 p1 = origin + b*dir;
//        ElVisFloat s0 = EvaluateFieldCuda(elementId, elementTypeId, p0);
//        ElVisFloat s1 = EvaluateFieldCuda(elementId, elementTypeId, p1);
//        ElVis::Interval<ElVisFloat> range;
//        EstimateRangeCuda(elementId, elementTypeId, fieldId, p0, p1, range);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Range of scalar field is (%2.10f, %2.10f)\n", range.GetLow(), range.GetHigh());
//        }

//        bool densityContainsBreakpoints = transferFunction->RangeContainsAtLeastOneBreakpoint(eDensity, range);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Density contains breakpoints %d.\n",densityContainsBreakpoints ? 1 : 0 );
//        }
//        if( !densityContainsBreakpoints )
//        {
//            // No breakpoints.  If 0 at the endpoints, then 0 everywhere.
//            if( transferFunction->Sample(eDensity, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eDensity, s1) == MAKE_FLOAT(0.0) )
//            {
//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("Density is identically 0.\n");
//                }
//                // Case 1
//                return;
//            }
//        }

//        // At this point we know that there is some non-0 density along the segment.
//        // Check if the color is identically 0.  If so, we can just integrate the
//        // density.
//        bool colorContainsBreakpoints = transferFunction->ColorContainsAtLeastOneBreakpoint(range);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Color contains breakpoints %d.\n",colorContainsBreakpoints ? 1 : 0 );
//        }
//        if( !colorContainsBreakpoints )
//        {
//            if( transferFunction->Sample(eRed, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eRed, s1) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eGreen, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eGreen, s1) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eBlue, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eBlue, s1) == MAKE_FLOAT(0.0) )
//            {
//                // Case 2 - Integrate density only.
//                if( densityContainsBreakpoints )
//                {
//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Integrate density alone using adaptive trapezoidal.\n");
//                    }

//                    // Case 2.1
//                    // Integrate density using adaptive trapezoidal.
//                }
//                else
//                {
//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Integrate density alone using gauss-kronrod.\n");
//                    }
//                    // Case 2.2
//                    // Integrate density using gauss-kronrod.
//                }
//                return;
//            }
//        }

//        // Case 3: Everything.  Both density and color contribute.
//        //    2.1.  No breakpoints in either.
//        //    2.2.  Color breakpoints, density no
//        //    2.3.  Color no breakpoints, density yes.
//        //    3.5  Color and density have breakpoints.
//        if( colorContainsBreakpoints )
//        {
//            // Need adaptive trapezoidal for the outer integral.  So evalute the density over the entire range,
//            // then do adaptive on the outer, sampling the inner function.
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Adaptive trapezoidal for the outer, stack backed for inner on [%2.10f, %2.10f].\n", a, b);
//            }
//            FieldEvaluator f;
//            f.Origin = origin;
//            f.Direction = dir;
//            f.ElementId = elementId;
//            f.ElementType = elementTypeId;
//            f.FieldId = fieldId;

//            bool reachedMax = false;
//            ElVisFloat maxValue = transferFunction->GetMaxValue(eDensity, range);
//            ElVisFloat estimate = maxValue*(b-a);
//            maxValue = MAKE_FLOAT(1.0);
//            estimate = MAKE_FLOAT(1.0);

//            InnerIntegralFunctor innerIntegralFunc;
//            innerIntegralFunc.transferFunction = transferFunction;

//            ReentrantAdaptiveTrapezoidal<ElVisFloat, 21> innerIntegralApproximation;
//            innerIntegralApproximation.Initialize(a, b, innerIntegralFunc, f, epsilon, estimate, maxValue, reachedMax, traceEnabled);

//            OuterIntegralIntegrand outerIntegralRedFunc;
//            outerIntegralRedFunc.channel = eRed;
//            outerIntegralRedFunc.innerIntegralApproximation = &innerIntegralApproximation;
//            outerIntegralRedFunc.transferFunction = transferFunction;

//            OuterIntegralIntegrand outerIntegralGreenFunc;
//            outerIntegralGreenFunc.channel = eGreen;
//            outerIntegralGreenFunc.innerIntegralApproximation = &innerIntegralApproximation;
//            outerIntegralGreenFunc.transferFunction = transferFunction;

//            OuterIntegralIntegrand outerIntegralBlueFunc;
//            outerIntegralBlueFunc.channel = eBlue;
//            outerIntegralBlueFunc.innerIntegralApproximation = &innerIntegralApproximation;
//            outerIntegralBlueFunc.transferFunction = transferFunction;

//            IntegrationStatus innerIntegrationStatus = ePartial;
//            int loopGuard = 0;
//            do
//            {
//                innerIntegrationStatus = innerIntegralApproximation.ContinueIntegration(innerIntegralFunc, f, epsilon, estimate, maxValue, reachedMax, traceEnabled);
//                ++loopGuard;

//                Interval<ElVisFloat> validDomain = innerIntegralApproximation.ValidDomain();
//                ReentrantAdaptiveTrapezoidal<ElVisFloat, 21> redApproximation;

//                if(traceEnabled)
//                {
//                    ELVIS_PRINTF("Finished evaluating the interval [%2.10f, %2.10f]\n", validDomain.GetLow(), validDomain.GetHigh());
//                }
//                redApproximation.Initialize(validDomain.GetLow(), validDomain.GetHigh(), outerIntegralRedFunc, f, epsilon, estimate, maxValue, reachedMax, false);

//                int redLoopGuard = 0;
//                IntegrationStatus redStatus = ePartial;
//                do
//                {
//                    ++redLoopGuard;
//                    redApproximation.ContinueIntegration(outerIntegralRedFunc, f, epsilon, estimate, maxValue, reachedMax, false);
//                    color.x += redApproximation.OverallValue();
//                }
//                while( redStatus != eFinished && redLoopGuard < 10);


//                ReentrantAdaptiveTrapezoidal<ElVisFloat, 21> greenApproximation;
//                greenApproximation.Initialize(validDomain.GetLow(), validDomain.GetHigh(), outerIntegralGreenFunc, f, epsilon, estimate, maxValue, reachedMax, false);

//                int greenLoopGuard = 0;
//                IntegrationStatus greenStatus = ePartial;
//                do
//                {
//                    ++greenLoopGuard;
//                    greenApproximation.ContinueIntegration(outerIntegralGreenFunc, f, epsilon, estimate, maxValue, reachedMax, false);
//                    color.y += greenApproximation.OverallValue();
//                }
//                while( greenStatus != eFinished && greenLoopGuard < 10);


//                ReentrantAdaptiveTrapezoidal<ElVisFloat, 21> blueApproximation;
//                blueApproximation.Initialize(validDomain.GetLow(), validDomain.GetHigh(), outerIntegralBlueFunc, f, epsilon, estimate, maxValue, reachedMax, false);

//                int blueLoopGuard = 0;
//                IntegrationStatus blueStatus = ePartial;
//                do
//                {
//                    ++blueLoopGuard;
//                    blueApproximation.ContinueIntegration(outerIntegralBlueFunc, f, epsilon, estimate, maxValue, reachedMax, false);
//                    color.z += blueApproximation.OverallValue();
//                }
//                while( blueStatus != eFinished && blueLoopGuard < 10);


//            }
//            while( innerIntegrationStatus != eFinished && loopGuard < 10 );
//            accumulatedDensity += innerIntegralApproximation.OverallValue();

////            if( traceEnabled )
////            {
////                ELVIS_PRINTF("################## Density\n");
////            }
////            innerIntegralApproximation.Integrate(a, b, transferFunction, eDensity, f, epsilon, estimate, maxValue, reachedMax, traceEnabled);

////            ElVisFloat redMax = transferFunction->GetMaxValue(eRed, range);
////            OuterIntegralAdaptiveTrapezoidal<ElVisFloat, 10> redIntegral;
////            if( traceEnabled )
////            {
////                ELVIS_PRINTF("################## Red\n");
////            }
////            redIntegral.Integrate(a, b, transferFunction, eRed, f, innerIntegralApproximation, epsilon, maxValue*redMax*(b-a), maxValue*redMax, reachedMax, traceEnabled, accumulatedDensity);
////            color.x += redIntegral.OverallValue();

////            ElVisFloat greenMax = transferFunction->GetMaxValue(eGreen, range);
////            OuterIntegralAdaptiveTrapezoidal<ElVisFloat, 10> greenIntegral;
////            if( traceEnabled )
////            {
////                ELVIS_PRINTF("################## Green\n");
////            }
////            greenIntegral.Integrate(a, b, transferFunction, eGreen, f, innerIntegralApproximation, epsilon, maxValue*greenMax*(b-a), maxValue*greenMax, reachedMax, traceEnabled, accumulatedDensity);
////            color.y += greenIntegral.OverallValue();

////            ElVisFloat blueMax = transferFunction->GetMaxValue(eBlue, range);
////            OuterIntegralAdaptiveTrapezoidal<ElVisFloat, 10> blueIntegral;
////            if( traceEnabled )
////            {
////                ELVIS_PRINTF("################## Blue\n");
////            }
////            blueIntegral.Integrate(a, b, transferFunction, eBlue, f, innerIntegralApproximation, epsilon, maxValue*blueMax*(b-a), maxValue*blueMax, reachedMax, traceEnabled, accumulatedDensity);
////            color.z += blueIntegral.OverallValue();

////            accumulatedDensity += innerIntegralApproximation.OverallValue();



//        }
//        else
//        {
//            // Color doesn't have breakpoints, so the main integral can be evaluated with Gauss-Kronrod.
//            // We'll do adaptive trapezoidal in the density, adding on to the existing integral as we sample
//            // the gauss-kronrod points.  This way we don't have to keep the adaptive structure around.
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Gauss-Kronrod outer, adaptive trapezoidal inner..\n");
//            }
//        }

//        densityAccumulator[segmentIndex] = accumulatedDensity;
//        colorAccumulator[segmentIndex] = color;
    }


    template<unsigned int block_size>
    __device__ void PrefixSumTrapezoidalIntegration(const ElVisFloat& initialValue, volatile ElVisFloat* __restrict__ input, volatile ElVisFloat* __restrict__ output, ElVisFloat h, bool traceEnabled)
    {
        ElVisFloat val = MAKE_FLOAT(.5)*input[threadIdx.x];
        output[threadIdx.x] = val;
        __syncthreads();

        if( threadIdx.x == 0 )
        {
            output[0] = MAKE_FLOAT(0.0);
        }

        if( traceEnabled )
        {
            ELVIS_PRINTF("Input Array Values: (");
            for(unsigned int i = 0; i < block_size; ++i)
            {
                ELVIS_PRINTF("%f, ", input[i]);
            }
            ELVIS_PRINTF(")\n\n");

            ELVIS_PRINTF("Output Array Values: (");
            for(unsigned int i = 0; i < block_size; ++i)
            {
                ELVIS_PRINTF("%f, ", output[i]);
            }
            ELVIS_PRINTF(")\n\n");
        }

        if(block_size >   1) { if (threadIdx.x >=   1) { val = output[threadIdx.x -   1] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   2) { if (threadIdx.x >=   2) { val = output[threadIdx.x -   2] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   4) { if (threadIdx.x >=   4) { val = output[threadIdx.x -   4] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   8) { if (threadIdx.x >=   8) { val = output[threadIdx.x -   8] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  16) { if (threadIdx.x >=  16) { val = output[threadIdx.x -  16] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  32) { if (threadIdx.x >=  32) { val = output[threadIdx.x -  32] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  64) { if (threadIdx.x >=  64) { val = output[threadIdx.x -  64] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 128) { if (threadIdx.x >= 128) { val = output[threadIdx.x - 128] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 256) { if (threadIdx.x >= 256) { val = output[threadIdx.x - 256] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 512) { if (threadIdx.x >= 512) { val = output[threadIdx.x - 512] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        output[threadIdx.x] = initialValue + h*(output[threadIdx.x]);
        __syncthreads();

        if(traceEnabled)
        {
            ELVIS_PRINTF("Result Array Values: (");
            for(unsigned int i = 0; i < block_size; ++i)
            {
                ELVIS_PRINTF("%f, ", output[i]);
            }
            ELVIS_PRINTF(")\n\n");
        }
    }

    template<unsigned int block_size>
    __device__ void PrefixSum(volatile ElVisFloat* __restrict__ input, volatile ElVisFloat* __restrict__ output)
    {
        ElVisFloat val = input[threadIdx.x];
        output[threadIdx.x] = val;
        __syncthreads();

        if(block_size >   1) { if (threadIdx.x >=   1) { val = output[threadIdx.x -   1] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   2) { if (threadIdx.x >=   2) { val = output[threadIdx.x -   2] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   4) { if (threadIdx.x >=   4) { val = output[threadIdx.x -   4] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >   8) { if (threadIdx.x >=   8) { val = output[threadIdx.x -   8] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  16) { if (threadIdx.x >=  16) { val = output[threadIdx.x -  16] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  32) { if (threadIdx.x >=  32) { val = output[threadIdx.x -  32] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size >  64) { if (threadIdx.x >=  64) { val = output[threadIdx.x -  64] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 128) { if (threadIdx.x >= 128) { val = output[threadIdx.x - 128] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 256) { if (threadIdx.x >= 256) { val = output[threadIdx.x - 256] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
        if(block_size > 512) { if (threadIdx.x >= 512) { val = output[threadIdx.x - 512] + val; } __syncthreads(); output[threadIdx.x] = val; __syncthreads(); }
    }


    template<typename IntegrationType>
    __device__ void IntegrateDensityFunction()
    {
    }

    // Assumes a single warp per ray, and that each block only contains a single warp.
    extern "C" __global__ void IntegrateSegmentWarpPerSegment(ElVisFloat3 origin, const int* __restrict__ segmentElementId, const int* __restrict__ segmentElementType,
                                                                             const ElVisFloat3* __restrict__ segmentDirection,
                                                                             const ElVisFloat* __restrict__ segmentStart, const ElVisFloat* __restrict__ segmentEnd,
                                                                             int fieldId,
                                                                             TransferFunction* transferFunction, ElVisFloat epsilon, bool enableTrace,
                                                                             ElVisFloat* __restrict__ densityAccumulator, ElVisFloat3* __restrict__ colorAccumulator)
    {
//        __shared__ int2 trace;
//        trace = make_int2(512/2, 512/2);
//
//        // Step 1 - Categorize the field along the segment.
//        //
//        // Case 1: Total space skipping.  When the density function is identically 0 over the entire segment, then there is nothing to do.
//        // Case 2: Density only.  If the color components are identially 0, then we only need to integrate the density contribution.
//        //    2.1: No breakpoints.  Use gauss-kronrod on the entire interval.  The 7-15 rule is probably sufficient.
//        //    2.2: Breakpoints.  Adaptive trapezoidal.
//        // Case 3: Everything.  Both density and color contribute.
//        //    2.1.  No breakpoints in either.
//        //    2.2.  Color breakpoints, density no
//        //    2.3.  Color no breakpoints, density yes.
//        //    3.5  Color and density have breakpoints.
//        // First pass - do adaptive trapezoidal on the density because I know I can then evaluate at any point within my error
//        // budget.  Then I can do adaptive or gauss-kronrod on the main integral without a problem.
//
//        // If the color is doing gauss-kronrod, then I can incrementally evaluate the density using adaptive trapezoidal and don't need to keep the entire
//        // structure around.  It is only when the outer integral is adaptive that I need to do that (case 2.2).
//
//        // In cases 2 and 3, each component can be integrated differently based on breakpoints.
//
//        // Density integration
//
//        __shared__ uint2 pixel;
//        pixel.x = blockIdx.x;
//        pixel.y = blockIdx.y;
//
//        bool traceEnabled = (pixel.x == trace.x && pixel.y == trace.y && enableTrace && threadIdx.x == 0);
//
////        if( traceEnabled )
////        {
////            ELVIS_PRINTF("Esilon %2.10f\n", epsilon);
////        }
//        __shared__ uint2 screen;
//        screen.x = gridDim.x;
//        screen.y = gridDim.y;
//
//        __shared__ int segmentIndex;
//        segmentIndex = pixel.x + screen.x*pixel.y;
//        __shared__ ElVisFloat b;
//        b = segmentEnd[segmentIndex];
//        if( b < MAKE_FLOAT(0.0) )
//        {
//            return;
//        }
//
//        __shared__ int elementId;
//        elementId = segmentElementId[segmentIndex];
//        if( elementId == -1 )
//        {
//            return;
//        }
//
//        __shared__ int elementTypeId;
//        elementTypeId = segmentElementType[segmentIndex];
//        __shared__ ElVisFloat accumulatedDensity;
//        accumulatedDensity = densityAccumulator[segmentIndex];
//        __shared__ ElVisFloat3 color;
//        color = colorAccumulator[segmentIndex];
//        __shared__ ElVisFloat a;
//        a = segmentStart[segmentIndex];
//
//
//        __shared__ ElVisFloat3 dir;
//        dir = segmentDirection[segmentIndex];
//        __shared__ ElVisFloat d;
//        d = (b-a);
//
//        if( d == MAKE_FLOAT(0.0) )
//        {
//            return;
//        }
//
//        __shared__ ElVisFloat h;
//        h = d/31;
//
//        // First test for density identically 0.  This means the segment does not contribute at
//        // all to the integral and can be skipped.
//        __shared__ ElVisFloat3 p0;
//        p0 = origin + a*dir;
//        __shared__ ElVisFloat3 p1;
//        p1 = origin + b*dir;
//        ElVisFloat s0 = EvaluateFieldCuda(elementId, elementTypeId, p0);
//        ElVisFloat s1 = EvaluateFieldCuda(elementId, elementTypeId, p1);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("%f, %f\n", s0, s1);
//        }
//        ElVis::Interval<ElVisFloat> range;
//        EstimateRangeCuda(elementId, elementTypeId, p0, p1, range);
//
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Range of scalar field is (%2.10f, %2.10f)\n", range.GetLow(), range.GetHigh());
//        }
//
//        bool densityContainsBreakpoints = transferFunction->RangeContainsAtLeastOneBreakpoint(eDensity, range);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Density contains breakpoints %d.\n",densityContainsBreakpoints ? 1 : 0 );
//        }
//        if( !densityContainsBreakpoints )
//        {
//            // No breakpoints.  If 0 at the endpoints, then 0 everywhere.
//            if( transferFunction->Sample(eDensity, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eDensity, s1) == MAKE_FLOAT(0.0) )
//            {
////                if( traceEnabled )
////                {
////                    ELVIS_PRINTF("Density is identically 0.\n");
////                }
//                // Case 1
//                return;
//            }
//        }
//
//        bool colorContainsBreakpoints = transferFunction->ColorContainsAtLeastOneBreakpoint(range);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Color contains breakpoints %d.\n",colorContainsBreakpoints ? 1 : 0 );
//        }
//        if( !colorContainsBreakpoints )
//        {
//            if( transferFunction->Sample(eRed, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eRed, s1) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eGreen, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eGreen, s1) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eBlue, s0) == MAKE_FLOAT(0.0) &&
//                transferFunction->Sample(eBlue, s1) == MAKE_FLOAT(0.0) )
//            {
//                // Case 2 - Integrate density only.
//                if( densityContainsBreakpoints )
//                {
//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Integrate density alone using adaptive trapezoidal.\n");
//                    }
//
//                    // Case 2.1
//                    // Integrate density using adaptive trapezoidal.
//                }
//                else
//                {
//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Integrate density alone using gauss-kronrod.\n");
//                    }
//                    // Case 2.2
//                    // Integrate density using gauss-kronrod.
//                    //IntegrateDensityFunction<G7K15>();
//                }
//                return;
//            }
//        }
//
//
////        // Sample the field.
////        __shared__ ElVisFloat fieldSamples[32];
////        ElVisFloat3 p = origin + (a+threadIdx.x*h)*dir;
////        fieldSamples[threadIdx.x] = EvaluateFieldCuda(elementId, elementTypeId, p);
////        __syncthreads();
//
////        __shared__ ElVisFloat density[32];
////        density[threadIdx.x] = transferFunction->Sample(eDensity, fieldSamples[threadIdx.x]);
////        __syncthreads();
//
////        __shared__ ElVisFloat accumulatedDensityIntegration[32];
////        PrefixSumTrapezoidalIntegration<32>(accumulatedDensity, &density[0], &accumulatedDensityIntegration[0], h, traceEnabled);
////        __syncthreads();
//
////        __shared__ ElVisFloat red[32];
////        __shared__ ElVisFloat green[32];
////        __shared__ ElVisFloat blue[32];
//
////        ElVisFloat attenuation = Exp(-accumulatedDensityIntegration[threadIdx.x]);
////        red[threadIdx.x] = transferFunction->Sample(eRed, fieldSamples[threadIdx.x])*density[threadIdx.x]*attenuation;
////        green[threadIdx.x] = transferFunction->Sample(eGreen, fieldSamples[threadIdx.x])*density[threadIdx.x]*attenuation;;
////        blue[threadIdx.x] = transferFunction->Sample(eBlue, fieldSamples[threadIdx.x])*density[threadIdx.x]*attenuation;;
////        __syncthreads();
//
////        __shared__ ElVisFloat redIntegral[32];
////        __shared__ ElVisFloat greenIntegral[32];
////        __shared__ ElVisFloat blueIntegral[32];
//
////        PrefixSumTrapezoidalIntegration<32>(color.x, &red[0], &redIntegral[0], h, traceEnabled);
////        PrefixSumTrapezoidalIntegration<32>(color.y, &green[0], &greenIntegral[0], h, traceEnabled);
////        PrefixSumTrapezoidalIntegration<32>(color.z, &blue[0], &blueIntegral[0], h, traceEnabled);
//
////        if( threadIdx.x == 0 )
////        {
////            densityAccumulator[segmentIndex] = accumulatedDensityIntegration[31];
////            color.x += redIntegral[31];
////            color.y += greenIntegral[31];
////            color.z += blueIntegral[31];
//
////            colorAccumulator[segmentIndex] = color;
////        }

    }

//    Actual code
    extern "C" __global__ void
    //__launch_bounds__(32, 8)
    IntegrateSegmentSingleThreadPerRay(ElVisFloat3 origin, const int* __restrict__ segmentElementId, const int* __restrict__ segmentElementType,
                                                                  const ElVisFloat3* __restrict__ segmentDirection,
                                                                  const ElVisFloat* __restrict__ segmentStart, const ElVisFloat* __restrict__ segmentEnd,
                                                                  int fieldId,
                                                                  TransferFunction* transferFunction, ElVisFloat epsilon, ElVisFloat desiredH, bool enableTrace,
                                                                  int tracex, int tracey,
                                                                  int* numSamples, bool renderIntegrationType,
                                                                  bool enableEmptySpaceSkipping,
                                                                  ElVisFloat* __restrict__ densityAccumulator, ElVisFloat3* __restrict__ colorAccumulator)
    {
        int2 trace = make_int2(tracex, tracey);


        uint2 pixel;
        pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
        pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

        bool traceEnabled = (pixel.x == trace.x && pixel.y == trace.y && enableTrace);

        if( traceEnabled )
        {
            ELVIS_PRINTF("Esilon %2.10f\n", epsilon);
            ELVIS_PRINTF("Number of samples enabled %d\n", (numSamples ? 1: 0));
            if( numSamples )
            {
                ELVIS_PRINTF("Value of samples: %d\n", numSamples[0]);
            }
        }
        uint2 screen;
        screen.x = gridDim.x * blockDim.x;
        screen.y = gridDim.y * blockDim.y;

        int segmentIndex = pixel.x + screen.x*pixel.y;
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Segment index %d\n", segmentIndex);
//        }
        if( segmentEnd[segmentIndex] < MAKE_FLOAT(0.0) )
        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because ray has left volume based on segment end\n", segmentIndex);
//            }
            return;
        }

        int elementId = segmentElementId[segmentIndex];
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Element id %d\n", elementId);
//        }
        if( elementId == -1 )
        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because element id is 0\n", segmentIndex);
//            }
            return;
        }

        int elementTypeId = segmentElementType[segmentIndex];
        ElVisFloat accumulatedDensity = densityAccumulator[segmentIndex];
        ElVisFloat3 color = colorAccumulator[segmentIndex];
        ElVisFloat a = segmentStart[segmentIndex];
        ElVisFloat b = segmentEnd[segmentIndex];

        ElVisFloat3 dir = segmentDirection[segmentIndex];
        ElVisFloat d = (b-a);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Ray Direction (%2.10f, %2.10f, %2.10f), segment distance %2.10f\n", dir.x, dir.y, dir.z, d);
//        }
        if( d == MAKE_FLOAT(0.0) )
        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because d is 0\n", dir.x, dir.y, dir.z, d);
//            }
            return;
        }

        int n = Floor(d/desiredH);

        ElVisFloat h;

        if( n == 0 )
        {
            h = b-a;
            n = 1;
        }
        else
        {
            h= d/(ElVisFloat)(n);
        }



//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Total segment range: [%2.15f, %2.15f], segment Id %d\n", segmentStart[segmentIndex], segmentEnd[segmentIndex], segmentIndex);
//            ELVIS_PRINTF("D = %2.15f, H = %2.15f, N = %d\n", d, h, n);
//        }

        // First test for density identically 0.  This means the segment does not contribute at
        // all to the integral and can be skipped.
        ElVisFloat3 p0 = origin + a*dir;
        ElVisFloat3 p1 = origin + b*dir;
        ElVis::Interval<ElVisFloat> range;
        EstimateRangeCuda(elementId, elementTypeId, fieldId, p0, p1, range);

        if( traceEnabled )
        {
            ELVIS_PRINTF("Range of scalar field is (%2.10f, %2.10f)\n", range.GetLow(), range.GetHigh());
            ELVIS_PRINTF("Origin (%f, %f, %f)\n", origin.x, origin.y, origin.z);

            ELVIS_PRINTF("Direction (%f, %f, %f)\n", dir.x, dir.y, dir.z);
            ELVIS_PRINTF("Integration domain [%f, %f]\n", a, b);
        }

        bool densityContainsBreakpoints = transferFunction->RangeContainsAtLeastOneBreakpoint(eDensity, range);
        Interval<ElVisFloat> densityRange = transferFunction->Sample(eDensity, range);
        if( traceEnabled )
        {
            ELVIS_PRINTF("Density contains breakpoints %d.\n",densityContainsBreakpoints ? 1 : 0 );
            ELVIS_PRINTF("Density range (%f, %f).\n", densityRange.GetLow(), densityRange.GetHigh());
        }

        if( enableEmptySpaceSkipping )
        {
            if( densityRange.GetLow() == MAKE_FLOAT(0.0) &&
                densityRange.GetHigh() == MAKE_FLOAT(0.0) )
            {
                if( traceEnabled )
                {
                    ELVIS_PRINTF("Density is identically 0.\n");
                }

//                if( renderIntegrationType )
//                {
//                    colorAccumulator[segmentIndex].x += MAKE_FLOAT(20.0)/MAKE_FLOAT(255.0);
//                }
                return;
            }
        }

        // At this point we know that there is some non-0 density along the segment.
        // Check if the color is identically 0.  If so, we can just integrate the
        // density.
        bool colorContainsBreakpoints = transferFunction->ColorContainsAtLeastOneBreakpoint(range);
        Interval<ElVisFloat> redRange = transferFunction->Sample(eRed, range);
        Interval<ElVisFloat> greenRange = transferFunction->Sample(eGreen, range);
        Interval<ElVisFloat> blueRange = transferFunction->Sample(eBlue, range);

        Interval<ElVisFloat> totalColorRange;
        totalColorRange.Combine(redRange);
        totalColorRange.Combine(blueRange);
        totalColorRange.Combine(greenRange);

        if( traceEnabled )
        {
            ELVIS_PRINTF("Color contains breakpoints %d.\n",colorContainsBreakpoints ? 1 : 0 );
            ELVIS_PRINTF("Red range (%f, %f).\n", redRange.GetLow(), redRange.GetHigh());
            ELVIS_PRINTF("Green range (%f, %f).\n", greenRange.GetLow(), greenRange.GetHigh());
            ELVIS_PRINTF("Blue range (%f, %f).\n", blueRange.GetLow(), blueRange.GetHigh());
            ELVIS_PRINTF("Total Color range (%f, %f).\n", totalColorRange.GetLow(), blueRange.GetHigh());
        }

        // If the color does not contribute, then we can just integrate the density.
        FieldEvaluator f;
        f.Origin = origin;
        f.Direction = dir;
        f.ElementId = elementId;
        f.ElementType = elementTypeId;
        f.sampleCount = numSamples;
        f.FieldId = fieldId;

//        bool colorEmpty = totalColorRange.GetLow() == MAKE_FLOAT(0.0) && totalColorRange.GetHigh() == MAKE_FLOAT(0.0);
//        int doDensityOnly = __all(colorEmpty);
//        __syncthreads();

        if( totalColorRange.GetLow() == MAKE_FLOAT(0.0) && totalColorRange.GetHigh() == MAKE_FLOAT(0.0) )
        //if( doDensityOnly )
        {
            InnerIntegralFunctor innerIntegralFunc;
            innerIntegralFunc.transferFunction = transferFunction;

//            int doBreakpoints  = __any(densityContainsBreakpoints);
//            __syncthreads();

            // Case 2 - Integrate density only.
            if( densityContainsBreakpoints )
            //if( doBreakpoints )
            {
                if( traceEnabled )
                {
                    ELVIS_PRINTF("Integrate density alone using adaptive trapezoidal.\n");
                }

                ElVisFloat result = FieldTrapezoidalIntegration::CompositeIntegration(innerIntegralFunc, f, a, b, n, traceEnabled);

                accumulatedDensity += result;
                f.AdjustSampleCount(-1);
            }
            else
            {
                if( traceEnabled )
                {
                    ELVIS_PRINTF("Integrate density alone using gauss-kronrod.\n");
                }

                ElVisFloat errorEstimate = MAKE_FLOAT(0.0);
                ElVisFloat result = SingleThreadGaussKronrod<G7K15>::Integrate<ElVisFloat>(innerIntegralFunc, a, b, f, errorEstimate, traceEnabled);
                accumulatedDensity += result;

//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("[%d, %d] - GK Density (%f, %f) - [%f, %f].\n", pixel.x, pixel.y, a, b, range.GetLow(), range.GetHigh());
//                    //ELVIS_PRINTF("G7K15 Density over segment %f with error %f\n", result, errorEstimate);
//                }
            }
        }
        else
        {
//            int doColorContainsBreakpoints = __any(colorContainsBreakpoints);
//            __syncthreads();

            // Color Contributes.
            // Case 3: Everything.  Both density and color contribute.
            //    2.1.  No breakpoints in either.
            //    2.2.  Color breakpoints, density no
            //    2.3.  Color no breakpoints, density yes.
            //    3.5  Color and density have breakpoints.
            if( colorContainsBreakpoints )
            //if( doColorContainsBreakpoints )
            {
                // Do trapezoidal for outer and inner in lockstep.
                if( traceEnabled )
                {
                    ELVIS_PRINTF("Trapezoidal for outer and inner.\n");
                }

//                if( renderIntegrationType )
//                {
//                    colorAccumulator[segmentIndex].y += MAKE_FLOAT(20.0)/MAKE_FLOAT(255.0);
//                    return;
//                }


                ElVisFloat s0 = f(a);
                ElVisFloat3 color0 = transferFunction->SampleColor(s0);
                ElVisFloat d0 = transferFunction->Sample(eDensity, s0);
                ElVisFloat atten = expf(-accumulatedDensity);
                color += h*MAKE_FLOAT(.5)*color0*d0*atten;

                for(int i = 1; i < n; ++i)
                {
                    ElVisFloat t = a + i*h;
                    ElVisFloat sample = f(t);
//                    if( traceEnabled )
//                    {
//                        ElVisFloat3 tempPoint = origin + t*dir;
//                        ELVIS_PRINTF("Sample at %f (%f, %f, %f) = %f\n", t, tempPoint.x, tempPoint.y, tempPoint.z, sample);
//                    }
                    ElVisFloat d1 = transferFunction->Sample(eDensity, sample);

                    accumulatedDensity += MAKE_FLOAT(.5)*h*(d0+d1);

//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Density = %f\n", d1);
//                    }

                    ElVisFloat3 colorSample = transferFunction->SampleColor(sample);
                    ElVisFloat atten = expf(-accumulatedDensity);

                    color += h*colorSample*d1*atten;

//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Density = %f, accumulated density = %f\n", d1, accumulatedDensity);
//                        ELVIS_PRINTF("Color Samples = (%f, %f, %f), Accumulated Color = (%f, %f, %f)\n", colorSample.x, colorSample.y, colorSample.z, color.x, color.y, color.z);
//                    }
                    d0 = d1;
                }

                ElVisFloat sn = f(b);
                ElVisFloat3 colorn = transferFunction->SampleColor(sn);
                ElVisFloat dn = transferFunction->Sample(eDensity, sn);
                accumulatedDensity += MAKE_FLOAT(.5)*h*(d0+dn);
                atten = expf(-accumulatedDensity);
                color += h*MAKE_FLOAT(.5)*colorn*dn*atten;

//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("Final Sample %f, Final Density Sample %f, Final Color Sample (%f, %f, %f)\n", sn, dn, colorn.x, colorn.y, colorn.z);
//                }
                f.AdjustSampleCount(-1);

            }
            else
            {
                // Color doesn't have breakpoints, so the main integral can be evaluated with Gauss-Kronrod.
                // We'll do adaptive trapezoidal in the density, adding on to the existing integral as we sample
                // the gauss-kronrod points.  This way we don't have to keep the adaptive structure around.
                if( traceEnabled )
                {
                    ELVIS_PRINTF("Gauss-Kronrod outer, Trapezoidal inner.\n");
                }

//                if( renderIntegrationType )
//                {
//                    colorAccumulator[segmentIndex].z += MAKE_FLOAT(20.0)/MAKE_FLOAT(255.0);
//                    return;
//                }

                OuterIntegralIntegrandWithInnerTrapezoidal outerIntegrand;
                outerIntegrand.accumulatedDensity = accumulatedDensity;
                outerIntegrand.field = &f;
                outerIntegrand.innerH = h;
                outerIntegrand.innerT = a;
                outerIntegrand.transferFunction = transferFunction;

                InnerIntegralFunctor innerIntegrand;
                innerIntegrand.transferFunction = transferFunction;
                outerIntegrand.innerIntegral = &innerIntegrand;

//                if( traceEnabled )
//                {
//                    ELVIS_PRINTF("Start GK with incoming density %2.15f\n", outerIntegrand.accumulatedDensity);
//                }

                ElVisFloat3 errorEstimate;
                ElVisFloat3 colorContribution = SingleThreadGaussKronrod<G7K15>::Integrate<ElVisFloat3>(outerIntegrand, a, b, f, errorEstimate, traceEnabled);

                // TODO - need to finish the density contribution for the space between the last sample and the end of the interval.

//                if( traceEnabled )
//                {
//                    ElVisFloat testDensity = FieldTrapezoidalIntegration::CompositeIntegration(innerIntegrand, f, a, b, n, traceEnabled);

//                    ELVIS_PRINTF("After running GK, the incoming color is (%2.15f, %2.15f, %2.15f), the color contribution is (%2.15f, %2.15f, %2.15f), and density contribution is %2.15f (test density is %2.15f) \n",
//                           color.x, color.y, color.z, colorContribution.x, colorContribution.y, colorContribution.z, outerIntegrand.accumulatedDensity, testDensity);

//                }
                color += colorContribution;
                accumulatedDensity = outerIntegrand.accumulatedDensity;

            }
        }

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Final density %2.15f\n", accumulatedDensity);
//            ELVIS_PRINTF("Final color is (%2.15f, %2.15f, %2.15f)\n", color.x, color.y, color.z);
//            if( numSamples )
//            {
//                ELVIS_PRINTF("Value of samples: %d\n", numSamples[0]);
//            }
//        }
        densityAccumulator[segmentIndex] = accumulatedDensity;
        colorAccumulator[segmentIndex] = color;
    }


    // Force GK/Trap
    extern "C" __global__ void
    //__launch_bounds__(32, 8)
    GKOnly(ElVisFloat3 origin, const int* __restrict__ segmentElementId, const int* __restrict__ segmentElementType,
                                                                  const ElVisFloat3* __restrict__ segmentDirection,
                                                                  const ElVisFloat* __restrict__ segmentStart, const ElVisFloat* __restrict__ segmentEnd,
                                                                  TransferFunction* transferFunction, ElVisFloat epsilon, ElVisFloat desiredH, bool enableTrace,
                                                                  int tracex, int tracey,
                                                                  int* numSamples, bool renderIntegrationType,
                                                                  bool enableEmptySpaceSkipping,
                                                                  ElVisFloat* __restrict__ densityAccumulator, ElVisFloat3* __restrict__ colorAccumulator)
    {
//        int2 trace = make_int2(tracex, tracey);


//        uint2 pixel;
//        pixel.x = blockIdx.x * blockDim.x + threadIdx.x;
//        pixel.y = blockIdx.y * blockDim.y + threadIdx.y;

//        bool traceEnabled = (pixel.x == trace.x && pixel.y == trace.y && enableTrace);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Esilon %2.10f\n", epsilon);
//            ELVIS_PRINTF("Number of samples enabled %d\n", (numSamples ? 1: 0));
//            if( numSamples )
//            {
//                ELVIS_PRINTF("Value of samples: %d\n", numSamples[0]);
//            }
//        }
//        uint2 screen;
//        screen.x = gridDim.x * blockDim.x;
//        screen.y = gridDim.y * blockDim.y;

//        int segmentIndex = pixel.x + screen.x*pixel.y;
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Segment index %d\n", segmentIndex);
//        }
//        if( segmentEnd[segmentIndex] < MAKE_FLOAT(0.0) )
//        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because ray has left volume based on segment end\n", segmentIndex);
//            }
//            return;
//        }

//        int elementId = segmentElementId[segmentIndex];
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Element id %d\n", elementId);
//        }
//        if( elementId == -1 )
//        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because element id is 0\n", segmentIndex);
//            }
//            return;
//        }

//        int elementTypeId = segmentElementType[segmentIndex];
//        ElVisFloat accumulatedDensity = densityAccumulator[segmentIndex];
//        ElVisFloat3 color = colorAccumulator[segmentIndex];
//        ElVisFloat a = segmentStart[segmentIndex];
//        ElVisFloat b = segmentEnd[segmentIndex];

//        ElVisFloat3 dir = segmentDirection[segmentIndex];
//        ElVisFloat d = (b-a);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Ray Direction (%2.10f, %2.10f, %2.10f), segment distance %2.10f\n", dir.x, dir.y, dir.z, d);
//        }
//        if( d == MAKE_FLOAT(0.0) )
//        {
//            if( traceEnabled )
//            {
//                ELVIS_PRINTF("Exiting because d is 0\n", dir.x, dir.y, dir.z, d);
//            }
//            return;
//        }

//        int n = Floor(d/desiredH);

//        ElVisFloat h;

//        if( n == 0 )
//        {
//            h = b-a;
//            n = 1;
//        }
//        else
//        {
//            h= d/(ElVisFloat)(n);
//        }



//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Total segment range: [%2.15f, %2.15f], segment Id %d\n", segmentStart[segmentIndex], segmentEnd[segmentIndex], segmentIndex);
//            ELVIS_PRINTF("D = %2.15f, H = %2.15f, N = %d\n", d, h, n);
//        }

//        // First test for density identically 0.  This means the segment does not contribute at
//        // all to the integral and can be skipped.
//        ElVisFloat3 p0 = origin + a*dir;
//        ElVisFloat3 p1 = origin + b*dir;
//        ElVisFloat s0 = EvaluateFieldCuda(elementId, elementTypeId, p0);
//        ElVisFloat s1 = EvaluateFieldCuda(elementId, elementTypeId, p1);
//        ElVis::Interval<ElVisFloat> range;
//        EstimateRangeCuda(elementId, elementTypeId, p0, p1, range);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Range of scalar field is (%2.10f, %2.10f)\n", range.GetLow(), range.GetHigh());
//            ELVIS_PRINTF("Origin (%f, %f, %f)\n", origin.x, origin.y, origin.z);

//            ELVIS_PRINTF("Direction (%f, %f, %f)\n", dir.x, dir.y, dir.z);
//            ELVIS_PRINTF("Integration domain [%f, %f]\n", a, b);
//        }

//        bool densityContainsBreakpoints = transferFunction->RangeContainsAtLeastOneBreakpoint(eDensity, range);
//        Interval<ElVisFloat> densityRange = transferFunction->Sample(eDensity, range);
//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Density contains breakpoints %d.\n",densityContainsBreakpoints ? 1 : 0 );
//            ELVIS_PRINTF("Density range (%f, %f).\n", densityRange.GetLow(), densityRange.GetHigh());
//        }

////        if( enableEmptySpaceSkipping )
////        {
////            if( densityRange.GetLow() == MAKE_FLOAT(0.0) &&
////                densityRange.GetHigh() == MAKE_FLOAT(0.0) )
////            {
////                if( traceEnabled )
////                {
////                    ELVIS_PRINTF("Density is identically 0.\n");
////                }

////                if( renderIntegrationType )
////                {
////                    colorAccumulator[segmentIndex].x += MAKE_FLOAT(1.0)/MAKE_FLOAT(255.0);
////                }
////                return;
////            }
////        }

//        // At this point we know that there is some non-0 density along the segment.
//        // Check if the color is identically 0.  If so, we can just integrate the
//        // density.
//        bool colorContainsBreakpoints = transferFunction->ColorContainsAtLeastOneBreakpoint(range);
//        Interval<ElVisFloat> redRange = transferFunction->Sample(eRed, range);
//        Interval<ElVisFloat> greenRange = transferFunction->Sample(eGreen, range);
//        Interval<ElVisFloat> blueRange = transferFunction->Sample(eBlue, range);

//        Interval<ElVisFloat> totalColorRange;
//        totalColorRange.Combine(redRange);
//        totalColorRange.Combine(blueRange);
//        totalColorRange.Combine(greenRange);

//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Color contains breakpoints %d.\n",colorContainsBreakpoints ? 1 : 0 );
//            ELVIS_PRINTF("Red range (%f, %f).\n", redRange.GetLow(), redRange.GetHigh());
//            ELVIS_PRINTF("Green range (%f, %f).\n", greenRange.GetLow(), greenRange.GetHigh());
//            ELVIS_PRINTF("Blue range (%f, %f).\n", blueRange.GetLow(), blueRange.GetHigh());
//            ELVIS_PRINTF("Total Color range (%f, %f).\n", totalColorRange.GetLow(), blueRange.GetHigh());
//        }

//        // If the color does not contribute, then we can just integrate the density.
//        FieldEvaluator f;
//        f.Origin = origin;
//        f.Direction = dir;
//        f.ElementId = elementId;
//        f.ElementType = elementTypeId;
//        f.sampleCount = numSamples;
//        f.FieldId = fieldId;

//                {
//                    ELVIS_PRINTF("[%d, %d] - GK Colro - Trap Density.\n", pixel.x, pixel.y);
//                    // Color doesn't have breakpoints, so the main integral can be evaluated with Gauss-Kronrod.
//                    // We'll do adaptive trapezoidal in the density, adding on to the existing integral as we sample
//                    // the gauss-kronrod points.  This way we don't have to keep the adaptive structure around.
//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Gauss-Kronrod outer, Trapezoidal inner.\n");
//                    }

//                    if( renderIntegrationType )
//                    {
//                        colorAccumulator[segmentIndex].z += MAKE_FLOAT(1.0)/MAKE_FLOAT(255.0);
//                        return;
//                    }

//                    OuterIntegralIntegrandWithInnerTrapezoidal outerIntegrand;
//                    outerIntegrand.accumulatedDensity = accumulatedDensity;
//                    outerIntegrand.field = &f;
//                    outerIntegrand.innerH = h;
//                    outerIntegrand.innerT = a;
//                    outerIntegrand.transferFunction = transferFunction;

//                    InnerIntegralFunctor innerIntegrand;
//                    innerIntegrand.transferFunction = transferFunction;
//                    outerIntegrand.innerIntegral = &innerIntegrand;

//                    if( traceEnabled )
//                    {
//                        ELVIS_PRINTF("Start GK with incoming density %2.15f\n", outerIntegrand.accumulatedDensity);
//                    }

//                    ElVisFloat3 errorEstimate;
//                    ElVisFloat3 colorContribution = SingleThreadGaussKronrod<G7K15>::Integrate<ElVisFloat3>(outerIntegrand, a, b, f, errorEstimate, traceEnabled);

//                    // TODO - need to finish the density contribution for the space between the last sample and the end of the interval.

//                    if( traceEnabled )
//                    {
//                        ElVisFloat testDensity = FieldTrapezoidalIntegration::CompositeIntegration(innerIntegrand, f, a, b, n, traceEnabled);

//                        ELVIS_PRINTF("After running GK, the incoming color is (%2.15f, %2.15f, %2.15f), the color contribution is (%2.15f, %2.15f, %2.15f), and density contribution is %2.15f (test density is %2.15f) \n",
//                               color.x, color.y, color.z, colorContribution.x, colorContribution.y, colorContribution.z, outerIntegrand.accumulatedDensity, testDensity);

//                    }
//                    color += colorContribution;
//                    accumulatedDensity = outerIntegrand.accumulatedDensity;

//                }


//        if( traceEnabled )
//        {
//            ELVIS_PRINTF("Final density %2.15f\n", accumulatedDensity);
//            ELVIS_PRINTF("Final color is (%2.15f, %2.15f, %2.15f)\n", color.x, color.y, color.z);
//            if( numSamples )
//            {
//                ELVIS_PRINTF("Value of samples: %d\n", numSamples[0]);
//            }
//        }
//        densityAccumulator[segmentIndex] = accumulatedDensity;
//        colorAccumulator[segmentIndex] = color;
    }








}

#endif
