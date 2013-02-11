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

#ifndef ELVIS_VOLUME_RENDERING_ADAPTIVE_TRAPEZOIDAL_IMPL_CU
#define ELVIS_VOLUME_RENDERING_ADAPTIVE_TRAPEZOIDAL_IMPL_CU

#include <ElVis/Core/EvaluateTransferFunction.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/FieldEvaluator.cu>
#include <ElVis/Core/RunElementByElementVolumeRendering.cu>
#include <ElVis/Math/TrapezoidalIntegration.hpp>

rtBuffer<ElVisFloat3, 2> raw_color_buffer;
rtBuffer<uchar4, 2> color_buffer;

rtDeclareVariable(rtObject, Volume, , );

rtBuffer<ElVisFloat> OpacityBreakpoints;
rtBuffer<ElVisFloat2> OpacityTransferFunction;

rtBuffer<ElVisFloat> IntensityBreakpoints;
rtBuffer<ElVisFloat2> IntensityTransferFunction;

template<typename IntegratorType>
__device__ void RunElementByElementVolumeRendering()
{
    // volume rendering uses ray type 2.
    uint2 screen = color_buffer.size();
    optix::Ray initialRay = GeneratePrimaryRay(screen, 2, 1e-3f);

    float3 origin0 = initialRay.origin;
    float3 rayDirection = initialRay.direction;

    //    ElVisFloat3 black = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    //    ElVisFloat3 red = MakeFloat3(MAKE_FLOAT(1.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
    //    ElVisFloat3 green = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(1.0), MAKE_FLOAT(0.0));
    //    color_buffer[launch_index] = ConvertToColor(black);

    VolumeRenderingPayload payload0;
    payload0.FoundIntersection = 0;

    optix::Ray ray0 = optix::make_Ray(initialRay.origin, initialRay.direction, 2, 1e-3, RT_DEFAULT_MAX);
    rtTrace(Volume, ray0, payload0);

    if( payload0.FoundIntersection == 0 )
    {
        ElVisFloat3 black = MakeFloat3(MAKE_FLOAT(0.0), MAKE_FLOAT(0.0), MAKE_FLOAT(0.0));
        color_buffer[launch_index] = ConvertToColor(black);
        return;
    }

    ElVisFloat finalColor = MAKE_FLOAT(0.0);
    ElVisFloat curOpacity = MAKE_FLOAT(0.0);
    bool done = false;
    int iter = 0;
    while(!done)
    {
        // Cast a second ray to find where we leave the current element.
        VolumeRenderingPayload payload1;
        payload1.FoundIntersection = 0;

        ElVisFloat3 origin1 = origin0 + payload0.IntersectionT*rayDirection;
        optix::Ray ray1 = optix::make_Ray(origin1, rayDirection, 2, 1e-3, RT_DEFAULT_MAX);
        rtTrace(Volume, ray1, payload1);

        if( payload1.FoundIntersection == 0 )
        {
            // This happens when we exit the volume.
            done = true;
            continue;
        }

        // Ideally, we could use the element id from the intersection tests, but it is difficult to know which
        // element we intersected.  In the worst case, the element from payload0 could be the previous element
        // and the element from payload1 is the next element
        // TODO - Investigate ray normals to figure out a more consistent marching.
        // FindElement is yet another ay, which isn't particularly cheap.
        // Even better - adjacency information from element to element.  From within an element, I can find the
        // exit face, then query the next element.
        int actualElementId = -1;
        int actualElementType = -1;
        bool found = FindElement(origin1 + (MAKE_FLOAT(.5)*payload1.IntersectionT)*rayDirection, actualElementId, actualElementType);


        if( found )
        {
            // Test the scalar range of the field.  This is ideal for detecting regions that
            // do not contribute because the transfer function is 0.
            //ElVisFloat3 p0 = origin1;
            ElVisFloat3 p1 = origin1 + payload1.IntersectionT*rayDirection;
            ElVis::Interval<ElVisFloat> estimatedRange;

            EstimateRange(actualElementId, actualElementType, origin1, p1, estimatedRange);

            // Does it matter where we declare this?
            EvaluateTransferFunction opacityTransfer(&(OpacityBreakpoints[0]), &(OpacityTransferFunction[0]), OpacityBreakpoints.size());


            //// Even if color doesn't intersect the scalar range, I need to update alpha.
            bool intervalIsEmpty = !opacityTransfer.IntersectsRange(estimatedRange);

            if( !intervalIsEmpty )
            {
                /////////////////////////////////////////////////
                // Integrate this ray segment.
                ////////////////////////////////////////////////
                FieldEvaluator f;
                f.Origin = origin1;
                f.Direction = rayDirection;
                f.ElementId = actualElementId;
                f.ElementType = actualElementType;

                EvaluateTransferFunction colorTransfer(&(IntensityBreakpoints[0]), &(IntensityTransferFunction[0]), IntensityBreakpoints.size());

                IntegratorType::Integrate(f, colorTransfer, opacityTransfer,
                    payload1.IntersectionT,
                    curOpacity,
                    finalColor);
            }
        }

        // Reset for next iteration.
        payload0.IntersectionT = payload1.IntersectionT;
        payload0.FoundIntersection = 0;
        payload0.ElementId = payload1.ElementId;
        payload0.ElementTypeId = payload1.ElementTypeId;
        origin0 = origin1;

        // For development debugging only to make sure I don't get into an infinite loop.
        done = iter > 50;
        ++iter;
    }


    ElVisFloat3 color = MakeFloat3(finalColor, finalColor, finalColor);
    color_buffer[launch_index] = ConvertToColor(color);
}

struct StackPoint
{
    template<typename ColorTransferFuncType, typename DensityTransferFunc, typename InnerIntegralFuncType,
        typename FieldFunc>
    __device__ __forceinline__
    void Evaluate(const ColorTransferFuncType& colorFunc, const DensityTransferFunc& densityFunc,
                  const ElVisFloat& initialDensity,
                  const InnerIntegralFuncType& innerIntegral,
                  const FieldFunc& fieldFunc)
    {
        ElVisFloat s = MAKE_FLOAT(0.0);
        //fieldFunc(TVal, s, Point);
        //ElVisFloat s = fieldFunc(TVal);
        //F = colorFunc(s)*densityFunc(s)*expf(- (initialDensity + innerIntegral.SampleInnerIntegral(TVal, s, densityFunc)));
    }

    __device__ __forceinline__  void Reset()
    {
        TVal = MAKE_FLOAT(1e30);
    }

    __device__ __forceinline__  bool IsUninitialized() const
    {
        return TVal == MAKE_FLOAT(1e30);
    }

    __device__ __forceinline__  StackPoint& operator=(const StackPoint& rhs)
    {
        Point = rhs.Point;
        TVal = rhs.TVal;
        F = rhs.F;
        return *this;
    }

    TensorPoint Point;
    ElVisFloat TVal;
    ElVisFloat F;
};




struct StackEntry
{
    __device__ __forceinline__  void CalculateMidpointT()
    {
        Mid().TVal = Left().TVal + (Right().TVal - Left().TVal)/2.0;
    }

    __device__ __forceinline__  void SetT(const ElVisFloat& t0, const ElVisFloat& t1)
    {
        Left().TVal = t0;
        Right().TVal = t1;
        CalculateMidpointT();
    }

    __device__ __forceinline__  ElVisFloat GetH() const
    {
        return Right().TVal - Left().TVal;
    }

    template<typename ColorTransferFunc, typename DensityTransferFunc, typename InnerIntegralFuncType, typename FieldFunc>
    __device__ __forceinline__  void EvaluateAll(
        const ColorTransferFunc& colorFunc,
        const DensityTransferFunc& densityFunc,
        const ElVisFloat& initialDensity,
        const InnerIntegralFuncType& innerIntegralFunc,
        const FieldFunc& fieldFunc)
    {
        for(unsigned int i = 0; i < 3; ++i)
        {
            points[i].Evaluate(colorFunc, densityFunc, initialDensity, innerIntegralFunc, fieldFunc);
        }
    }

    __device__ __forceinline__  StackPoint& Left()  { return points[0]; }
    __device__ __forceinline__  StackPoint& Mid()  { return points[1]; }
    __device__ __forceinline__  StackPoint& Right() { return points[2]; }

    __device__ __forceinline__  const StackPoint& Left() const  { return points[0]; }
    __device__ __forceinline__  const StackPoint& Mid() const  { return points[1]; }
    __device__ __forceinline__  const StackPoint& Right() const { return points[2]; }

    StackPoint points[3];
};




struct AdaptiveTrapezoidalIntegration
{
    /// \param integrationEndpoint - The endpoint of the integration along a ray
    ///                              that starts at 0.
    /// \param curOpacity - Input/Ouptut.  On input, it is the current accumulated
    ///                     density.  On output, the contribution of this segment is
    ///                     added in.
    __device__ static void Integrate(const FieldEvaluator& f,
            const EvaluateTransferFunction& color,
            const EvaluateTransferFunction& tau,
            ElVisFloat integrationEndpoint,
            ElVisFloat& curDensity,
            ElVisFloat& finalColor)
    {
        // Check to see the range of the field along the ray.
        // Check each breakpoint to see if they are in the range.
        // If it is in the range, evaluate both sides.  If either side is non-0, then we go
        // ahead and do the integration.

        //
        // Step 1 - Integrate the density function along the ray to create an integral function
        // that can be sampled when evaluating the outer integral.
        ElVisFloat epsilon = MAKE_FLOAT(1e-5);
        //ElVis::Math::IterativeAdaptiveTrapezoidalIntegralFunction<ElVisFloat, 8> innerIntegralApproximation;
        //ElVis::Math::IterativeAdaptiveTrapezoidalIntegralFunctionInterleaved<ElVisFloat, 8> innerIntegralApproximation;
        ElVis::Math::IterativeAdaptiveTrapezoidalIntegralFunctionStackVersion<ElVisFloat, 6> innerIntegralApproximation;
        //ElVis::Math::IterativeAdaptiveTrapezoidalIntegralFunctionRegistersOnly<ElVisFloat, 8> innerIntegralApproximation;
        //ElVis::Math::IterativeAdaptiveTrapezoidalIntegralFunctionUsingThreeSeparateArrays<ElVisFloat, 8> innerIntegralApproximation(&TBuffer[0], &FBuffer[0], &IBuffer[0]);


        innerIntegralApproximation.Integrate(MAKE_FLOAT(0.0), integrationEndpoint, tau, f, epsilon);
        //curDensity = innerIntegralApproximation.OverallValue();
        //finalColor = innerIntegralApproximation.OverallValue();


        ElVisFloat accumulatedIntegral = 0.0;
        const unsigned int maxRecursion = 8;
        StackEntry stack[maxRecursion];

        stack[0].SetT(MAKE_FLOAT(0.0), integrationEndpoint);
        stack[0].EvaluateAll(color, tau, curDensity, innerIntegralApproximation, f);

        stack[1].Left() = stack[0].Left();
        stack[1].Mid().Reset();
        stack[1].Right() = stack[0].Mid();

        unsigned int minimumDepth = 7;

        int i = 1;
//        while( i > 0 )
//        {
//            if( stack[i].Mid().IsUninitialized() )
//            {
//                bool needToSubdivide = false;

//                stack[i].CalculateMidpointT();
//                stack[i].Mid().Evaluate(color, tau, curDensity, innerIntegralApproximation, f);

//                if( i < minimumDepth )
//                {
//                    needToSubdivide = true;
//                }
//                else
//                {
//                    ElVisFloat I0 = stack[i].GetH()/MAKE_FLOAT(2.0) * (stack[i].Left().F + stack[i].Right().F);
//                    ElVisFloat I1 = stack[i].GetH()/MAKE_FLOAT(4.0) * (stack[i].Left().F + 2.0*stack[i].Mid().F + stack[i].Right().F);
//                    ElVisFloat diff = fabs(I0-I1)/MAKE_FLOAT(6.0);

//                    // Determine if we need to subdivide, or if this estimate is good enough.
//                    // A key optimization is that there will be ranges in the transfer function where
//                    // the contribution is exactly 0 for the range.  This makes adaptive quadrature
//                    // a little tricky.
//                    //
//                    // Test for a breakpoint in each subinterval.  If there is a breakpoint, then we
//                    // will force a subdivision.  This will force the algorithm to recurse to the maximum
//                    // recursion limit, which we should evaluate at some point.
//                    ElVis::Interval<ElVisFloat> leftRange = EstimateRangeFromTensorPoints(f.ElementId, f.ElementType, stack[i].Left().Point, stack[i].Mid().Point);
//                    ElVis::Interval<ElVisFloat> rightRange = EstimateRangeFromTensorPoints(f.ElementId, f.ElementType, stack[i].Mid().Point, stack[i].Right().Point);

//                    bool leftRangeContainsBreakpoint = tau.ScalarRangeContainsBreakpoint(leftRange);
//                    bool rightRangeContainsBreakpoint = tau.ScalarRangeContainsBreakpoint(rightRange);

//                    bool rangeTest = leftRangeContainsBreakpoint;
//                    rangeTest |= rightRangeContainsBreakpoint;


//                    //
//                    // If there are no breakpoints in either interval, then we can use the error
//                    // estimator to determine if we recurse.  By putting the breakpoint check first,
//                    // this will automatically stop if we are in a 0 area of the transfer function.
//                    ElVisFloat localEpsilon = epsilon * (stack[i].GetH()/stack[0].GetH());
//                    bool epsilonTest = (diff > localEpsilon);
//                    epsilonTest &= (i < maxRecursion-1);

//                    needToSubdivide = rangeTest;
//                    needToSubdivide |= epsilonTest;

//                    if( launch_index.x == 700 && launch_index.y == 350 )
//                    {
//                        rtPrintf("Left Scalar Range: (%f, %f), Right Scalar Range: (%f, %f)\n", leftRange.GetLow(), leftRange.GetHigh(), rightRange.GetLow(), rightRange.GetHigh());
//                        rtPrintf("Range Test (%d), Epsilon Test (%d), Need to subdivide (%d)\n", rangeTest, epsilonTest, needToSubdivide);
//                    }
//                }

//                if( needToSubdivide )
//                {
//                    stack[i+1].Left() = stack[i].Left();
//                    stack[i+1].Mid().Reset();
//                    stack[i+1].Right() = stack[i].Mid();
//                    i = i + 1;
//                }
//                else
//                {
//                    ElVisFloat h = stack[i].GetH()/MAKE_FLOAT(4.0);
//                    ElVisFloat mid_f = stack[i].Mid().F;
//                    ElVisFloat right_f = stack[i].Right().F;

//                    ElVisFloat leftContribution = h * (stack[i].Left().F + mid_f);
//                    ElVisFloat rightContribution = h * (mid_f + right_f);

//                    accumulatedIntegral += leftContribution + rightContribution;
//                }
//            }
//            else
//            {
//                if( stack[i].Right().TVal == stack[i-1].Mid().TVal )
//                {
//                    // We just finished traversing the left side, now go to
//                    // the right.
//                    stack[i].Left() = stack[i-1].Mid();
//                    stack[i].Mid().Reset();
//                    stack[i].Right() = stack[i-1].Right();
//                }
//                else
//                {
//                    // We finished this branch.  Remove it and go up to
//                    // the next one.
//                    i = i-1;
//                }
//            }
//        }

        accumulatedIntegral = fminf(MAKE_FLOAT(1.0), accumulatedIntegral);

        finalColor += accumulatedIntegral;
        curDensity += innerIntegralApproximation.OverallValue();
    }

};


RT_PROGRAM void VolumeRendererAdaptiveTrapezoidalIntegration()
{
    RunElementByElementVolumeRendering<AdaptiveTrapezoidalIntegration>();
}

#endif
