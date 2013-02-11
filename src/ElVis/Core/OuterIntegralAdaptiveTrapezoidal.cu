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

#ifndef ELVIS_CORE_OUTER_INTEGRAL_ADAPTIVE_TRAPEZOIDAL_CU
#define ELVIS_CORE_OUTER_INTEGRAL_ADAPTIVE_TRAPEZOIDAL_CU

namespace ElVis
{
//    template<typename T, unsigned int n>
//    struct OuterIntegralAdaptiveTrapezoidal
//    {
//        public:
//            struct StackPoint
//            {
//                template<typename FieldFunc, typename InnerIntegralType>
//                __device__
//                void Evaluate(const TransferFunction* transferFunction,
//                              TransferFunctionChannel channel,
//                              const FieldFunc& fieldFunc,
//                              const InnerIntegralType& innerIntegral,
//                              const T& accumulatedDensity, bool traceEnabled)
//                {
//                    T s = fieldFunc(TVal);
//                    T density = transferFunction->Sample(eDensity, s);
//                    T color = transferFunction->Sample(channel, s);
//                    T attenuation =::exp(-(accumulatedDensity+innerIntegral.SampleInnerIntegral(TVal, s, eDensity, transferFunction)));

//                    F = density*color*attenuation;

//                    if( traceEnabled )
//                    {
//                        printf("f(%f) = (%f) * (%f) * (%f)\n", TVal, color, density, attenuation);
//                    }
//                }

//                __device__ void Reset()
//                {
//                    TVal = MAKE_FLOAT(1e30);
//                }

//                __device__ bool IsUninitialized() const
//                {
//                    return TVal == MAKE_FLOAT(1e30);
//                }

//                __device__ StackPoint& operator=(const StackPoint& rhs)
//                {
//                    TVal = rhs.TVal;
//                    F = rhs.F;
//                    return *this;
//                }

//                T TVal;
//                T F;
//            };




//            struct StackEntry
//            {
//                __device__ void CalculateMidpointT()
//                {
//                    Mid().TVal = Left().TVal + (Right().TVal - Left().TVal)/2.0;
//                }

//                __device__ void SetT(const T& t0, const T& t1)
//                {
//                    Left().TVal = t0;
//                    Right().TVal = t1;
//                    CalculateMidpointT();
//                }

//                __device__ T GetH() const
//                {
//                    return Right().TVal - Left().TVal;
//                }

//                template<typename FieldFunc, typename InnerIntegralType>
//                __device__ void EvaluateAll(const TransferFunction* densityFunc,
//                                           TransferFunctionChannel channel,
//                                           const FieldFunc& fieldFunc,
//                                           const InnerIntegralType& innerIntegral,
//                                           const T& accumulatedDensity, bool traceEnabled)
//                {
//                    for(unsigned int i = 0; i < 3; ++i)
//                    {
//                        points[i].Evaluate(densityFunc, channel, fieldFunc, innerIntegral, accumulatedDensity, traceEnabled);
//                    }
//                }

//                __device__ StackPoint& Left()  { return points[0]; }
//                __device__ StackPoint& Mid()  { return points[1]; }
//                __device__ StackPoint& Right() { return points[2]; }

//                __device__ const StackPoint& Left() const  { return points[0]; }
//                __device__ const StackPoint& Mid() const  { return points[1]; }
//                __device__ const StackPoint& Right() const { return points[2]; }

//                StackPoint points[3];
//            };


//            template<typename FieldFunctionType, typename InnerIntegralType>
//            __device__ void Integrate(const T& t0, const T& t1, const TransferFunction* transferFunction,
//                                     TransferFunctionChannel channel,
//                                     const FieldFunctionType& fieldFunction,
//                                     const InnerIntegralType& innerIntegral,
//                                     const T& globalEpsilon,
//                                     const T& globalIntegralEstimate,
//                                     const T& maxFunctionValue, bool& reachedMaxRecursion,
//                                     bool traceEnabled,
//                                     const T& accumulatedDensity)
//            {
//                if( traceEnabled )
//                {
//                    printf("Global Epsilon %f, globalIntegralEstimate %f, maxValue %f\n", globalEpsilon, globalIntegralEstimate, maxFunctionValue);
//                }


//                const unsigned int maxRecursion = n;
//                reachedMaxRecursion = false;
//                StackEntry stack[maxRecursion];


//                stack[0].SetT(t0, t1);
//                stack[0].EvaluateAll(transferFunction, channel, fieldFunction, innerIntegral, accumulatedDensity, traceEnabled);

//                stack[1].Left() = stack[0].Left();
//                stack[1].Mid().Reset();
//                stack[1].Right() = stack[0].Mid();

//                unsigned int minimumDepth = 2;

//                int i = 1;
//                t[0] = t0;
//                f[0] = stack[0].Left().F;
//                I[0] = 0.0;
//                adaptiveIndex = 0;
//                while( i > 0 )
//                {
//                    reachedMaxRecursion |= (i == maxRecursion-1);
//                    if( stack[i].Mid().IsUninitialized() )
//                    {
//                        bool needToSubdivide = false;

//                        stack[i].CalculateMidpointT();
//                        stack[i].Mid().Evaluate(transferFunction, channel, fieldFunction, innerIntegral, accumulatedDensity, traceEnabled);

//                        if( i < minimumDepth )
//                        {
//                            needToSubdivide = true;
//                        }
//                        else
//                        {
//                            T I0 = stack[i].GetH()/MAKE_FLOAT(2.0) * (stack[i].Left().F + stack[i].Right().F);
//                            T I1 = stack[i].GetH()/MAKE_FLOAT(4.0) * (stack[i].Left().F + 2.0*stack[i].Mid().F + stack[i].Right().F);
//                            T localEpsilon = globalEpsilon*globalIntegralEstimate * (stack[i].GetH()/stack[0].GetH());

//                            if( traceEnabled )
//                            {
//                                printf("Level %d, Interval (%f, %f, %f), values (%f, %f, %f) I0 = %f, I1 = %f, localEpsilon = %f\n", i, stack[i].Left().TVal, stack[i].Mid().TVal, stack[i].Right().TVal,
//                                       stack[i].Left().F, stack[i].Mid().F, stack[i].Right().F, I0, I1, localEpsilon);
//                            }

//                            ElVisFloat h = stack[i].GetH()/MAKE_FLOAT(2.0);

//                            if( stack[i].Left().F == MAKE_FLOAT(0.0) &&
//                                stack[i].Mid().F == MAKE_FLOAT(0.0) &&
//                                stack[i].Right().F == MAKE_FLOAT(0.0) )
//                            {
//                                ElVis::Interval<ElVisFloat> range = fieldFunction.EstimateRange(stack[i].Left().TVal, stack[i].Right().TVal);

//                                ElVisFloat maxValue = transferFunction->GetMaxValue(channel, range)*transferFunction->GetMaxValue(eDensity, range);
//                                T maxSegmentError = (maxFunctionValue*h)/globalIntegralEstimate;
//                                T updatedSegmentError = (maxValue*h)/globalIntegralEstimate;

//                                if( traceEnabled )
//                                {
//                                    printf("All 3 values are 0.  Scalar range is (%f, %f), maxSegmentError %f, updatedSegmentError %f\n", range.GetLow(), range.GetHigh(),
//                                           maxSegmentError, updatedSegmentError);
//                                }

//                                if( updatedSegmentError > localEpsilon && i < maxRecursion-1 )
//                                {
//                                    needToSubdivide = true;
//                                }
//                            }
//                            else if( stack[i].Left().F == MAKE_FLOAT(0.0) ||
//                                stack[i].Mid().F == MAKE_FLOAT(0.0) ||
//                                stack[i].Right().F == MAKE_FLOAT(0.0) )
//                            {
//                                // If any of the samples are 0, then we know there is a breakpoint somewhere and we should subdivide.
//                                T maxSegmentError = (maxFunctionValue*h)/globalIntegralEstimate;

//                                ElVis::Interval<ElVisFloat> range = fieldFunction.EstimateRange(stack[i].Left().TVal, stack[i].Right().TVal);

//                                ElVisFloat maxValue = transferFunction->GetMaxValue(channel, range)*transferFunction->GetMaxValue(eDensity, range);
//                                T updatedSegmentError = (maxValue*h)/globalIntegralEstimate;

//                                if( traceEnabled )
//                                {
//                                    printf("At least one value is 0.  Scalar range is (%f, %f), maxSegmentError %f, updatedSegmentError %f\n", range.GetLow(), range.GetHigh(),
//                                           maxSegmentError, updatedSegmentError);
//                                }

//                                if( traceEnabled )
//                                {
//                                    printf("One of the samples is 0, maxSegmentError = %f, localEpsilon = %f\n", maxSegmentError, localEpsilon);
//                                }
//                                if(updatedSegmentError > localEpsilon && i < maxRecursion-1 )
//                                {
//                                    needToSubdivide = true;
//                                }
//                            }
//                            else
//                            {
//                                T errorEstimate = fabs(I0-I1)/globalIntegralEstimate;
//                                if( traceEnabled )
//                                {
//                                    printf("No samples 0, errorEstimate = %f, localEpsilon = %f\n", errorEstimate, localEpsilon);
//                                }
//                                if( errorEstimate > localEpsilon && i < maxRecursion-1 )
//                                {
//                                    needToSubdivide = true;
//                                }
//                            }
//                        }

//                        if( traceEnabled )
//                        {
//                            printf("Subdividing = %d\n", needToSubdivide? 1 : 0);
//                        }

//                        if( needToSubdivide )
//                        {
//                            stack[i+1].Left() = stack[i].Left();
//                            stack[i+1].Mid().Reset();
//                            stack[i+1].Right() = stack[i].Mid();
//                            i = i + 1;
//                        }
//                        else
//                        {
//                            T prevValue = I[adaptiveIndex];
//                            T h = stack[i].GetH()/MAKE_FLOAT(4.0);
//                            T mid_f = stack[i].Mid().F;
//                            T right_f = stack[i].Right().F;

//                            t[adaptiveIndex+1] = stack[i].Mid().TVal;
//                            t[adaptiveIndex+2] = stack[i].Right().TVal;
//                            f[adaptiveIndex+1] = mid_f;
//                            f[adaptiveIndex+2] = right_f;

//                            T leftContribution = h * (stack[i].Left().F + mid_f);
//                            T rightContribution = h * (mid_f + right_f);

//                            I[adaptiveIndex+1] = prevValue + leftContribution;
//                            I[adaptiveIndex+2] = prevValue + leftContribution+rightContribution;
//                            if( traceEnabled )
//                            {
//                                printf("Integral Value at %f = %f\n", t[adaptiveIndex+1], I[adaptiveIndex+1]);
//                                printf("Integral Value at %f = %f\n", t[adaptiveIndex+2], I[adaptiveIndex+2]);
//                            }

//                            adaptiveIndex += 2;
//                        }
//                    }
//                    else
//                    {
//                        if( stack[i].Right().TVal == stack[i-1].Mid().TVal )
//                        {
//                            // We just finished traversing the left side, now go to
//                            // the right.
//                            stack[i].Left() = stack[i-1].Mid();
//                            stack[i].Mid().Reset();
//                            stack[i].Right() = stack[i-1].Right();
//                        }
//                        else
//                        {
//                            // We finished this branch.  Remove it and go up to
//                            // the next one.
//                            i = i-1;
//                        }
//                    }
//                }

//            }

//            template<typename DensityFuncType>
//            __device__ T SampleInnerIntegral(T t_i, T sample, TransferFunctionChannel channel, const TransferFunction* densityFunc) const
//            {
//                if( t_i < t[0] ||
//                    t_i > t[adaptiveIndex] )
//                {
//                    return MAKE_FLOAT(0.0);
//                }

//                if( t_i == t[0] ) return MAKE_FLOAT(0.0);
//                if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

//                const T* a = &(t[0]);
//                const T* b = &(t[adaptiveIndex]);
//                while(b-a > 1 )
//                {
//                    const T* mid = (b-a)/2 + a;
//                    if( *mid == t_i )
//                    {
//                        return I[mid-a];
//                    }
//                    if( t_i < *mid )
//                    {
//                        b = mid;
//                    }
//                    else
//                    {
//                        a = mid;
//                    }
//                }

//                T baseline = I[a-t];
//                T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc->Sample(channel, sample));
//                return baseline+segment;
//            }

//            __device__ T OverallValue() const
//            {
//                return I[adaptiveIndex];
//            }

//            static const unsigned int arraySize = (0x01 << n) + 1;

//            T t[arraySize];
//            T f[arraySize];
//            T I[arraySize];
//        private:
//            unsigned int adaptiveIndex;
//    };


}


#endif




