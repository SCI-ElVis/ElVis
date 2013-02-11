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

#ifndef ELVIS_CORE_REENTRANT_ADAPTIVE_TRAPEZOIDAL_CU
#define ELVIS_CORE_REENTRANT_ADAPTIVE_TRAPEZOIDAL_CU

namespace ElVis
{
//    template<typename T, unsigned int n>
//    struct ReentrantAdaptiveTrapezoidal
//    {
//        public:
//            struct StackPoint
//            {
//                template<typename IntegrandType, typename FieldFunc>
//                __device__
//                void Evaluate(const IntegrandType& integrand,
//                    const FieldFunc& fieldFunc)
//                {
//                    T s = fieldFunc(TVal);
//                    F = integrand(TVal, s);
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
//                    Mid().TVal = (Right().TVal + Left().TVal)*MAKE_FLOAT(.5);
//                }

//                __device__ void SetT(const T& t0, const T& t1)
//                {
//                    Left().TVal = t0;
//                    Right().TVal = t1;
//                    CalculateMidpointT();
//                }

//                __device__ void CreateFromRight(const StackEntry& s)
//                {
//                    points[0] = s.points[1];
//                    points[2] = s.points[2];
//                    CalculateMidpointT();
//                }

//                __device__ void CreateFromLeft(const StackEntry& s)
//                {
//                    points[0] = s.points[0];
//                    points[2] = s.points[1];
//                    CalculateMidpointT();
//                }

//                __device__ T GetH() const
//                {
//                    return Right().TVal - Left().TVal;
//                }

//                template<typename IntegrandType, typename FieldFunc>
//                __device__ void EvaluateAll(const IntegrandType& integrand,
//                    const FieldFunc& fieldFunc)
//                {
//                    for(unsigned int i = 0; i < 3; ++i)
//                    {
//                        points[i].Evaluate(integrand, fieldFunc);
//                    }
//                }

//                __device__ StackEntry& operator=(const StackEntry& rhs)
//                {
//                    points[0] = rhs.points[0];
//                    points[1] = rhs.points[1];
//                    points[2] = rhs.points[2];
//                    return *this;
//                }

//                __device__ StackPoint& Left()  { return points[0]; }
//                __device__ StackPoint& Mid()  { return points[1]; }
//                __device__ StackPoint& Right() { return points[2]; }

//                __device__ const StackPoint& Left() const  { return points[0]; }
//                __device__ const StackPoint& Mid() const  { return points[1]; }
//                __device__ const StackPoint& Right() const { return points[2]; }

//                StackPoint points[3];
//            };



//            struct Stack
//            {
//                template<typename IntegrandType, typename FieldFunction>
//                __device__ void Initialize(const T& t0, const T& t1, const IntegrandType& integrand, const FieldFunction& fieldFunction)
//                {
//                    // Initialize the stack prior to execution.
//                    stack[0].SetT(t0, t1);
//                    stack[0].EvaluateAll(integrand, fieldFunction);

//                    curIndex = 0;
//                    baseH = t1-t0;
//                }

//                __device__ StackEntry Pop(bool traceEnabled)
//                {
//                    if( curIndex < 0 ) return stack[0];

//                    StackEntry result = stack[curIndex];
//                    curIndex -= 1;
//                    if( traceEnabled )
//                    {
//                        printf("After popping.\n");
//                        PrintStack(traceEnabled);
//                    }
//                    return result;
//                }

//                __device__ bool HasCapacity(int num)
//                {
//                    return curIndex + num < n;
//                }

//                __device__ bool Push(const StackEntry& s, bool traceEnabled)
//                {
//                    if( curIndex + 2 >= n )
//                    {
//                        if( traceEnabled )
//                        {
//                            printf("Attempting to push onto stack but not enough space.\n");
//                        }
//                        return false;
//                    }

//                    StackEntry right;
//                    right.CreateFromRight(s);

//                    StackEntry left;
//                    left.CreateFromLeft(s);

//                    if( right.Mid().TVal == right.Left().TVal ||
//                        right.Mid().TVal == right.Right().TVal ||
//                        left.Mid().TVal == left.Left().TVal ||
//                        left.Mid().TVal == left.Right().TVal )
//                    {
//                        return false;
//                    }

//                    stack[curIndex+1] = right;
//                    stack[curIndex+2] = left;
//                    if( traceEnabled )
//                    {
//                        printf("Pushing [%2.10f, %2.10f, %2.10f] with values (%2.10f, %2.10f, %2.10f) onto location %d\n", right.Left().TVal, right.Mid().TVal, right.Right().TVal, right.Left().F, right.Mid().F, right.Right().F, curIndex+1);
//                        printf("Pushing [%2.10f, %2.10f, %2.10f] with values (%2.10f, %2.10f, %2.10f) onto location %d\n", left.Left().TVal, left.Mid().TVal, left.Right().TVal, left.Left().F, left.Mid().F, left.Right().F,  curIndex+2);
//                    }

//                    curIndex += 2;
//                    PrintStack(traceEnabled);
//                    return true;
//                }

//                __device__ void PrintStack(bool traceEnabled)
//                {
//                    if( traceEnabled )
//                    {
//                        for(int i = 0; i <= curIndex; ++i)
//                        {
//                            printf("[%d] = (%2.10f, %2.10f, %2.10f) and values (%2.10f, %2.10f, %2.10f) \n", i, stack[i].Left().TVal, stack[i].Mid().TVal, stack[i].Right().TVal, stack[i].Left().F, stack[i].Mid().F, stack[i].Right().F);
//                        }
//                    }
//                }

//                __device__ T GetBaseH() { return baseH; }

//                __device__ StackEntry& Top() { return stack[curIndex]; }

//                __device__ bool Empty() { return curIndex == -1; }
//                __device__ int Depth() { return curIndex; }

//                StackEntry stack[n];
//                int curIndex;
//                ElVisFloat baseH;
//            };

//            template<typename IntegrandType, typename FieldFunctionType>
//            __device__ void Initialize(const T& t0, const T& t1, const IntegrandType& integrand,
//                const FieldFunctionType& fieldFunction, const T& globalEpsilon,
//                const T& globalIntegralEstimate,
//                                     const T& maxFunctionValue, bool& reachedMaxRecursion,
//                                     bool traceEnabled)
//            {
//                if( traceEnabled )
//                {
//                    printf("Initializing range [%2.10f, %2.10f]\n", t0, t1);
//                }
//                stack.Initialize(t0, t1, integrand, fieldFunction);

//                // Put it at the end, since the continue function will copy it to the beginning.
//                t[n-1] = t0;
//                f[n-1] = stack.Top().Left().F;
//                I[n-1] = 0.0;
//            }



//            template<typename IntegralFunc, typename FieldFunctionType>
//            __device__ IntegrationStatus ContinueIntegration(const IntegralFunc& integrand,
//                const FieldFunctionType& fieldFunction, const T& globalEpsilon,
//                const T& globalIntegralEstimate,
//                                     const T& maxFunctionValue, bool& reachedMaxRecursion,
//                                     bool traceEnabled)
//            {
//                if( traceEnabled )
//                {
//                    printf("Global Epsilon %2.10f, globalIntegralEstimate %2.10f, maxValue %2.10f\n", globalEpsilon, globalIntegralEstimate, maxFunctionValue);
//                }


//                reachedMaxRecursion = false;

//                unsigned int minimumDepth = 2;

//                t[0] = t[n-1];
//                f[0] = f[n-1];
//                I[0] = I[n-1];

//                if( traceEnabled )
//                {
//                    printf("########################################3 Restarting with t = %2.10f, f = %2.10f, I = %2.10f\n", t[0], f[0], I[0]);
//                }

//                int loopGuard = 0;
//                endIndex = 1;
//                while( !stack.Empty() && endIndex < n && loopGuard < 50)
//                {
//                    reachedMaxRecursion |= stack.HasCapacity(1);
//                    StackEntry curStack = stack.Pop(traceEnabled);

//                    bool needToSubdivide = false;

//                    curStack.CalculateMidpointT();
//                    curStack.Mid().Evaluate(integrand, fieldFunction);

//                    if( stack.Depth() < minimumDepth )
//                    {
//                        if( traceEnabled )
//                        {
//                            printf("Subdividing because of minimum depth.\n");
//                        }
//                        needToSubdivide = true;
//                    }
//                    else
//                    {
//                        ElVisFloat h2 = curStack.GetH()*MAKE_FLOAT(.5);
//                        ElVisFloat h4 = h2*MAKE_FLOAT(.5);

//                        if( h4 == MAKE_FLOAT(0.0) )
//                        {
//                            if( traceEnabled )
//                            {
//                                printf("Stopping subdivision because h is 0.\n");
//                            }
//                            //goto PushPop;
//                        }

//                        T localEpsilon = globalEpsilon* (curStack.GetH()/stack.GetBaseH());

//                        if( localEpsilon == MAKE_FLOAT(0.0) )
//                        {
//                            if( traceEnabled )
//                            {
//                                printf("Stopping subdivision because epsilon is 0.\n");
//                            }
//                            //goto PushPop;
//                        }

//                        if( h4 > MAKE_FLOAT(0.0) && localEpsilon > MAKE_FLOAT(0.0) )
//                        {
//                            T I0 = h2 * (curStack.Left().F + curStack.Right().F);
//                            T I1 = h4 * (curStack.Left().F + 2.0*curStack.Mid().F + curStack.Right().F);

//                            if( traceEnabled )
//                            {
//                                printf("Level %d, Interval (%2.10f, %2.10f, %2.10f), values (%2.10f, %2.10f, %2.10f) I0 = %2.10f, I1 = %2.10f, localEpsilon = %2.10f\n", stack.Depth(), curStack.Left().TVal, curStack.Mid().TVal, curStack.Right().TVal,
//                                       curStack.Left().F, curStack.Mid().F, curStack.Right().F, I0, I1, localEpsilon);
//                            }

//                            ElVisFloat h = curStack.GetH()/MAKE_FLOAT(2.0);

//                            bool rangeCheckEnabled = curStack.Left().F == MAKE_FLOAT(0.0);
//                            rangeCheckEnabled &= curStack.Mid().F == MAKE_FLOAT(0.0);
//                            rangeCheckEnabled &= curStack.Right().F == MAKE_FLOAT(0.0);


//                            if( rangeCheckEnabled )
//                            {
//                                // If any of the samples are 0, then we know there is a breakpoint somewhere and we should subdivide.
//                                T maxSegmentError = (maxFunctionValue*h)/globalIntegralEstimate;

//                                ElVis::Interval<ElVisFloat> range = fieldFunction.EstimateRange(curStack.Left().TVal, curStack.Right().TVal);

//                                ElVisFloat maxValue = integrand.GetMaxValue(range);
//                                T updatedSegmentError = (maxValue*h)/globalIntegralEstimate;

//                                if( traceEnabled )
//                                {
//                                    printf("At least one value is 0.  Scalar range is (%2.10f, %2.10f), maxSegmentError %2.10f, updatedSegmentError %2.10f\n", range.GetLow(), range.GetHigh(),
//                                           maxSegmentError, updatedSegmentError);
//                                }

//                                if( traceEnabled )
//                                {
//                                    printf("One of the samples is 0, maxSegmentError = %2.10f, localEpsilon = %2.10f\n", maxSegmentError, localEpsilon);
//                                }
//                                //needToSubdivide = updatedSegmentError > localEpsilon;
//                                if(updatedSegmentError > localEpsilon )
//                                {
//                                    needToSubdivide = true;
//                                }
//                            }
//                            else
//                            {
//                                T errorEstimate = fabs(I0-I1)/globalIntegralEstimate;
//                                if( traceEnabled )
//                                {
//                                    printf("No samples 0, errorEstimate = %2.10f, localEpsilon = %2.10f\n", errorEstimate, localEpsilon);
//                                }

//                                if( errorEstimate > localEpsilon )
//                                {
//                                    needToSubdivide = true;
//                                }
//                            }
//                        }
//                    }

//                    if( traceEnabled )
//                    {
//                        printf("Subdividing = %d\n", needToSubdivide? 1 : 0);
//                    }

//                    bool failedToSubdivide = true;
//                    if( needToSubdivide && stack.HasCapacity(2) )
//                    {
//                        // Push onto the stack
//                        failedToSubdivide = !stack.Push(curStack, traceEnabled);
//                    }

//                    if( failedToSubdivide )
//                    {
//                        // Update values and pop off the stack.
//                        T prevValue = I[endIndex-1];
//                        T h = curStack.GetH()/MAKE_FLOAT(4.0);
//                        T mid_f = curStack.Mid().F;
//                        T right_f = curStack.Right().F;

//                        t[endIndex] = curStack.Mid().TVal;
//                        t[endIndex+1] = curStack.Right().TVal;
//                        f[endIndex] = mid_f;
//                        f[endIndex+1] = right_f;

//                        T leftContribution = h * (curStack.Left().F + mid_f);
//                        T rightContribution = h * (mid_f + right_f);

//                        I[endIndex] = prevValue + leftContribution;
//                        I[endIndex+1] = prevValue + leftContribution+rightContribution;

//                        if( traceEnabled )
//                        {
//                            printf("prevValue %2.10f, h %2.10f, mid_f %2.10f, right_f %2.10f, leftContribution %2.10f, rightContribution %2.10f\n", prevValue, h, mid_f, right_f, leftContribution, rightContribution);
//                            printf("Integral Value at f(%2.10f) = %2.10f is %2.10f\n", t[endIndex], f[endIndex], I[endIndex]);
//                            printf("Integral Value at f(%2.10f) = %2.10f is %2.10f\n", t[endIndex+1], f[endIndex+1], I[endIndex+1]);
//                        }
//                        endIndex += 2;
//                    }
//                    loopGuard += 1;
//                }

//                if( stack.Empty() )
//                {
//                    if( traceEnabled )
//                    {
//                        printf("Stack is empty, end index %d, loopGuard %d.\n", endIndex, loopGuard);
//                    }
//                    return eFinished;
//                }
//                else
//                {
//                    if( traceEnabled )
//                    {
//                        printf("Stack is not empty end index %d, loopGuard %d.\n", endIndex, loopGuard);
//                    }
//                    return ePartial;
//                }
//            }

//            __device__ T SampleInnerIntegral(T t_i, T sample, TransferFunctionChannel channel, const TransferFunction* densityFunc) const
//            {
//                if( t_i < t[0] ||
//                    t_i > t[endIndex-1] )
//                {
//                    return MAKE_FLOAT(0.0);
//                }

//                if( t_i == t[0] ) return MAKE_FLOAT(0.0);
//                if( t_i == t[endIndex-1] ) return I[endIndex-1];

//                const T* a = &(t[0]);
//                const T* b = &(t[endIndex-1]);
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
//                return I[endIndex-1];
//            }


//            __device__ Interval<ElVisFloat> ValidDomain() const
//            {
//                Interval<ElVisFloat> result;
//                result.SetLow(t[0]);
//                result.SetHigh(t[endIndex-1]);
//                return result;
//            }

//            Stack stack;
//            T t[n];
//            T f[n];
//            T I[n];

//        private:
//            // During evaluation, always = n.
//            // On the last interval, can be less.
//            unsigned int endIndex;
//    };
}

#endif
