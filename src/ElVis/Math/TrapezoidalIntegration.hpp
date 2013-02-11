////////////////////////////////////////////////////////////////////////////////
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description:
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ELVIS_MATH_TRAPEZOIDAL_INTEGRATION_HPP
#define ELVIS_MATH_TRAPEZOIDAL_INTEGRATION_HPP

#include <ElVis/Core/Cuda.h>

#include <ElVis/Math/TrapezoidalIntegration.hpp>
#include <ElVis/Math/AdaptiveQuadrature.hpp>

#include <ElVis/Core/Interval.hpp>

#include <math.h>

#if defined(__CUDACC__)
#define RT_DEVICE __device__ __forceinline__
#else
#define RT_DEVICE
#endif


namespace ElVis
{
    using ::fabs;
    namespace Math
    {
        struct TrapezoidalIntegration
        {
            template<typename T, typename FuncType>
            ELVIS_DEVICE static T Integrate(const FuncType& f, const T& a, const T& b)
            {
                return (b-a)/2.0 * (f(a) + f(b));
            }

            // n is the number of intervals.
            // So evaluation is at n+1 points.
            template<typename T, typename FuncType>
            ELVIS_DEVICE static T CompositeIntegration(const FuncType& f, const T& a, const T& b, unsigned int n)
            {
                T h = (b-a)/(n);
                T result = 0.0;
                for(unsigned int i = 1; i < n; ++i)
                {
                    result += f(a+i*h);
                }
                result += .5*f(a) + .5*f(b);
                result*=h;
                return result;
            }
        };

        struct CorrectedTrapezoidalIntegration
        {
            template<typename T, typename FuncType, typename DerivType>
            static T Evaluate(const FuncType& f, const FuncType& deriv, const T& a, const T& b, unsigned int n)
            {
                T h = (b-a)/n;
                T result = 0.0;
                for(unsigned int i = 0; i < n; ++i)
                {
                    result += f(a+i*h)*h;
                }
                result += h*.5*f(a) + h*.5*f(b);

                result += h*h/12.0 * (deriv(a) - deriv(b));
                return result;
            }
        };



        struct AdaptiveTrapezoidalIntegration
        {


            template<typename T, typename FuncType>
            static T Integrate(const FuncType& f, const T& a, const T& b, const double& epsilon)
            {
                unsigned int n = 10;
                T h = (b-a)/n;
                T result = 0.0;
                for(unsigned int i = 0; i < n; ++i)
                {
                    unsigned int temp;
                    result += AdaptiveIntegrate(f, a+i*h, a+(i+1)*h, epsilon, temp);
                }
                return result;
            }

            template<typename T, typename FuncType>
            static T AdaptiveIntegrate(const FuncType& f, const T& a, const T& b, 
                const double& epsilon, 
                unsigned int& n)
            {
                // Each call required one more sample than already taken.  This code actually samples 
                // again, but for the sake of convergence analysis, we only count one more.
                ++n;
                T h = (b-a);
                T h2 = h/2.0;
                if( Width<T>::Evaluate(h) == 0.0 || Width<T>::Evaluate(h2) == 0.0 ) return 0.0;

                T midpoint = a + h2;

                T f0 = f(a);
                T f1 = f(a+h2);
                T f2 = f(b);

                T I1 = h/2.0 * (f0 + f2);
                T left = h2/2.0 * (f0+f1);
                T right = h2/2.0 * (f1+f2);
                T I2 = left + right;
                T errorEstimate = fabs(I1-I2)/6.0;

                bool needsSubdivision = false;
                Interval<T> range;

                // Relative error rather than absolute.
                bool localEpsilonTest = errorEstimate < epsilon*I1;
                if( errorEstimate > epsilon 
                    && epsilon > 1e-9 )
                {
                    needsSubdivision = true;
                }
                //else if( (f0==f1) && (f1==f2) )
                //{
                //    // For volume rendering, we'll often have regions of constant value.
                //    // Use interval arithmetic to see if it is constant over the entire 
                //    // region or not.

                //    // If h is small, then we can assume that the function is constant 
                //    // over the interval.
                //    if( h > 1e-4 )
                //    {
                //        // If h is large, we should check the range of the function.
                //        Interval<T> domain(a, b);
                //        range = f(domain);
                //        if( range.GetWidth() > 0.0 )
                //        {
                //            needsSubdivision = true;
                //        }
                //    }
                //}
                // If the three samples are the same then we have nothing to base 
                // our error estimate on, so we need to subdivide.  However, if 
                // we are in a constant section of the transfer function, then this 
                // is accurate, so we don't want to spend too long recursing.
                //if( (epsilon > 1e-9) &&
                //    ( (errorEstimate > epsilon) ||
                //      ( (f0==f1) && (f1==f2) && (h > 1e-3) )) 
                //   )
                if( needsSubdivision )
                {
                    T newEpsilon = epsilon/2.0;
                    return AdaptiveTrapezoidalIntegration::AdaptiveIntegrate(f, a, midpoint, newEpsilon, n) +
                        AdaptiveTrapezoidalIntegration::AdaptiveIntegrate(f, midpoint, b, newEpsilon, n);
                }
                else
                {
                    return I2;
                }
            }

            template<typename T, typename FuncType>
            static T AdaptiveIntegrate(const FuncType& f, const T& a, const T& b, 
                const double& epsilon, 
                const double& globalEpsilon,
                const T& globalEstimate,
                unsigned int& n)
            {
                // Each call required one more sample than already taken.  This code actually samples 
                // again, but for the sake of convergence analysis, we only count one more.
                ++n;
                T h = (b-a);
                T h2 = h/2.0;
                if( Width<T>::Evaluate(h) == 0.0 || Width<T>::Evaluate(h2) == 0.0 ) return 0.0;

                T midpoint = a + h2;

                if( midpoint <= a ||
                    midpoint >= b )
                {
                    return 0.0;
                }

                T f0 = f(a);
                T f1 = f(a+h2);
                T f2 = f(b);

                T I1 = h/2.0 * (f0 + f2);
                T left = h2/2.0 * (f0+f1);
                T right = h2/2.0 * (f1+f2);
                T I2 = left + right;
                T errorEstimate = fabs(I1-I2)/6.0;

                bool needsSubdivision = false;
                Interval<T> range;

                // Relative error rather than absolute.
                bool localEpsilonTest = (errorEstimate < epsilon*fabs(I1)) || (errorEstimate < epsilon);
                bool globalEpsilonTest = fabs(I2) < globalEpsilon*globalEstimate;
                if( !localEpsilonTest && !globalEpsilonTest )
                {
                    needsSubdivision = true;
                }
                //else if( (f0==f1) && (f1==f2) )
                //{
                //    // For volume rendering, we'll often have regions of constant value.
                //    // Use interval arithmetic to see if it is constant over the entire 
                //    // region or not.

                //    // If h is small, then we can assume that the function is constant 
                //    // over the interval.
                //    if( h > 1e-4 )
                //    {
                //        // If h is large, we should check the range of the function.
                //        Interval<T> domain(a, b);
                //        range = f(domain);
                //        if( range.GetWidth() > 0.0 )
                //        {
                //            needsSubdivision = true;
                //        }
                //    }
                //}
                // If the three samples are the same then we have nothing to base 
                // our error estimate on, so we need to subdivide.  However, if 
                // we are in a constant section of the transfer function, then this 
                // is accurate, so we don't want to spend too long recursing.
                //if( (epsilon > 1e-9) &&
                //    ( (errorEstimate > epsilon) ||
                //      ( (f0==f1) && (f1==f2) && (h > 1e-3) )) 
                //   )
                if( needsSubdivision )
                {
                    T newEpsilon = epsilon/2.0;
                    return AdaptiveTrapezoidalIntegration::AdaptiveIntegrate(f, a, midpoint, newEpsilon, globalEpsilon, globalEstimate, n) +
                        AdaptiveTrapezoidalIntegration::AdaptiveIntegrate(f, midpoint, b, newEpsilon, globalEpsilon, globalEstimate, n);
                }
                else
                {
                    return I2;
                }
            }
        };
   
        /// n - An upper bound on the number of recursive calls that can be made.  The higher n, the
        /// more memory this object will consume, the longer it will take to computer, and the
        /// more accurate the result.
        template<typename T, unsigned int n>
        struct IterativeAdaptiveTrapezoidalIntegralFunctionUsingThreeSeparateArrays
        {
            public:

                RT_DEVICE IterativeAdaptiveTrapezoidalIntegralFunctionUsingThreeSeparateArrays(T* tParam, T* fParam, T* IParam) :
                    t(tParam),
                    f(fParam),
                    I(IParam)
                {
                }

                template<typename TransferFunctionType, typename FieldFunctionType>
                RT_DEVICE void Integrate(T t0, T t1, const TransferFunctionType& transferFunction,
                    const FieldFunctionType& fieldFunction, const T epsilon)
                {
                    unsigned int curDepth = 1;

                    // The stride tells us both how wide the distance between samples in the
                    // array.
                    unsigned int nonAdaptiveIndex = 0;
                    adaptiveIndex = 0;

                    // The minimum number of subdivisions to perform before using
                    // adaptive quadrature.
                    const unsigned int minimumTreeDepth = 8;

                    unsigned int index[] = {0, (arraySize-1)/2, arraySize-1};
                    ElVisFloat temp[] = {t0, (t1-t0)/MAKE_FLOAT(2.0) + t0, t1};
                    for(unsigned int i = 0; i < 3; ++i)
                    {
                        t[index[i]] = temp[i];
                        f[index[i]] = transferFunction(fieldFunction(temp[i]));
                    }

                    // Do the initial sampling before starting the loop.
//                    t[0] = t0;
//                    f[0] = transferFunction(fieldFunction(t[0]));

//                    t[arraySize-1] = t1;
//                    f[arraySize-1] = transferFunction(fieldFunction(t[arraySize-1]));

                    T range = t1-t0;
//                    t[(arraySize-1)/2] = (t1-t0)/2.0 + t0;
//                    f[(arraySize-1)/2] = transferFunction(fieldFunction(t[(arraySize-1)/2]));

                    I[0] = MAKE_FLOAT(0.0);

                    bool done = false;
                    // Sample.  Incrementing by 1 is a depth-first traversal.
                    while(!done)
                    {
                        unsigned int stride = (0x01 << (n-curDepth));

                        // Sample the midpoint of this interval.
                        unsigned int nonAdaptiveMidpointIndex = nonAdaptiveIndex + stride/2;
                        unsigned int nonAdaptiveEndIndex = nonAdaptiveIndex + stride;
                        T h = t[nonAdaptiveIndex+stride] - t[nonAdaptiveIndex];
                        T localEpsilon = epsilon * (h/range);
                        t[nonAdaptiveMidpointIndex] = t[nonAdaptiveIndex] + h/MAKE_FLOAT(2.0);
                        f[nonAdaptiveMidpointIndex] = transferFunction(fieldFunction(t[nonAdaptiveMidpointIndex]));

                        bool needToSubdivideAgain = false;

                        T overallEstimate = h/MAKE_FLOAT(2.0) *(f[nonAdaptiveIndex] + f[nonAdaptiveEndIndex]);
                        T leftEstimate = h/MAKE_FLOAT(4.0) * (f[nonAdaptiveIndex] + f[nonAdaptiveMidpointIndex]);
                        T rightEstimate = h/MAKE_FLOAT(4.0) * (f[nonAdaptiveMidpointIndex] + f[nonAdaptiveEndIndex]);

                        I[nonAdaptiveMidpointIndex] = I[nonAdaptiveIndex] + leftEstimate;
                        I[nonAdaptiveEndIndex] = I[nonAdaptiveMidpointIndex] + rightEstimate;

                        // If the curDepth is n-1, then we have reached the end of the tree, so no subdivision
                        // is possible.
                        if( curDepth != (n-1) )
                        {
                            if( curDepth <= minimumTreeDepth )
                            {
                                needToSubdivideAgain = true;
                            }
                            else
                            {
                                if( fabsf(overallEstimate - (leftEstimate+rightEstimate))/MAKE_FLOAT(6.0) > localEpsilon &&
                                    localEpsilon > MAKE_FLOAT(1e-9) )
                                {
                                    needToSubdivideAgain = true;
                                }
                            }
                        }

                        // Algorithm to go to the next node in the graph.

                        if( needToSubdivideAgain )
                        {
                            // If we are subdividing, then the next iteration uses the same index,
                            // just with smaller interval.
                            adaptiveIndex = adaptiveIndex;
                            curDepth += 1;
                        }
                        else
                        {

                            // Update the adaptive values.
                            t[adaptiveIndex] = t[nonAdaptiveIndex];
                            t[adaptiveIndex + 1] = t[nonAdaptiveMidpointIndex];

                            f[adaptiveIndex] = f[nonAdaptiveIndex];
                            f[adaptiveIndex + 1] = f[nonAdaptiveMidpointIndex];

                            I[adaptiveIndex] = I[nonAdaptiveIndex];
                            I[adaptiveIndex + 1] = I[nonAdaptiveMidpointIndex];

                            // If we are not subdividing, then the value of the integral to this
                            // point has been calculated and we can move on.
                            adaptiveIndex += 2;
                            nonAdaptiveIndex += stride;

                            // Check possible strides, from the maximum to 2.
                            for(unsigned int j = n-1; j > 0; --j)
                            {
                                unsigned int checkStride = (0x01 << j);
                                if( nonAdaptiveIndex % checkStride == 0 )
                                {
                                    curDepth = n-j;
                                    break;
                                }
                            }

                            if( nonAdaptiveIndex == arraySize-1 )
                            {
                                t[adaptiveIndex] = t[nonAdaptiveEndIndex];
                                f[adaptiveIndex] = f[nonAdaptiveEndIndex];
                                I[adaptiveIndex] = I[nonAdaptiveEndIndex];
                                done = true;
                            }
                        }
                    }

                }

                template<typename DensityFuncType>
                RT_DEVICE T SampleInnerIntegral(T t_i, T sample, const DensityFuncType& densityFunc) const
                {
                    if( t_i < t[0] ||
                        t_i > t[adaptiveIndex] )
                    {
                        return MAKE_FLOAT(0.0);
                    }

                    if( t_i == t[0] ) return MAKE_FLOAT(0.0);
                    if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

                    const T* a = &(t[0]);
                    const T* b = &(t[adaptiveIndex]);
                    while(b-a > 1 )
                    {
                        const T* mid = (b-a)/MAKE_FLOAT(2.0) + a;
                        if( *mid == t_i )
                        {
                            return I[mid-a];
                        }
                        if( t_i < *mid )
                        {
                            b = mid;
                        }
                        else
                        {
                            a = mid;
                        }
                    }

                    T baseline = I[a-t];
                    T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc(sample));
                    return baseline+segment;
                }

                RT_DEVICE T OverallValue() const
                {
                    return I[adaptiveIndex];
                }

                static const unsigned int arraySize = (0x01 << n) + 1;

                T* t;
                T* f;
                T* I;
            private:
                unsigned int adaptiveIndex;
        };


        /// n - An upper bound on the number of recursive calls that can be made.  The higher n, the 
        /// more memory this object will consume, the longer it will take to computer, and the 
        /// more accurate the result.
        template<typename T, unsigned int n>
        struct IterativeAdaptiveTrapezoidalIntegralFunction
        {
            public:
                template<typename TransferFunctionType, typename FieldFunctionType>
                RT_DEVICE void Integrate(T t0, T t1, const TransferFunctionType& transferFunction,
                    const FieldFunctionType& fieldFunction, const T epsilon)
                {
                    unsigned int curDepth = 1;

                    // The stride tells us both how wide the distance between samples in the
                    // array.
                    unsigned int nonAdaptiveIndex = 0;
                    adaptiveIndex = 0;

                    // The minimum number of subdivisions to perform before using
                    // adaptive quadrature.
                    const unsigned int minimumTreeDepth = 8;

                    unsigned int index[] = {0, (arraySize-1)/2, arraySize-1};
                    ElVisFloat temp[] = {t0, (t1-t0)/MAKE_FLOAT(2.0) + t0, t1};
                    for(unsigned int i = 0; i < 3; ++i)
                    {
                        t[index[i]] = temp[i];
                        f[index[i]] = transferFunction(fieldFunction(temp[i]));
                    }

                    // Do the initial sampling before starting the loop.
//                    t[0] = t0;
//                    f[0] = transferFunction(fieldFunction(t[0]));

//                    t[arraySize-1] = t1;
//                    f[arraySize-1] = transferFunction(fieldFunction(t[arraySize-1]));

                    T range = t1-t0;
//                    t[(arraySize-1)/2] = (t1-t0)/2.0 + t0;
//                    f[(arraySize-1)/2] = transferFunction(fieldFunction(t[(arraySize-1)/2]));

                    I[0] = MAKE_FLOAT(0.0);

                    bool done = false;
                    // Sample.  Incrementing by 1 is a depth-first traversal.
                    while(!done)
                    {
                        unsigned int stride = (0x01 << (n-curDepth));

                        // Sample the midpoint of this interval.
                        unsigned int nonAdaptiveMidpointIndex = nonAdaptiveIndex + stride/2;
                        unsigned int nonAdaptiveEndIndex = nonAdaptiveIndex + stride;
                        T h = t[nonAdaptiveIndex+stride] - t[nonAdaptiveIndex];
                        T localEpsilon = epsilon * (h/range);
                        t[nonAdaptiveMidpointIndex] = t[nonAdaptiveIndex] + h/MAKE_FLOAT(2.0);
                        f[nonAdaptiveMidpointIndex] = transferFunction(fieldFunction(t[nonAdaptiveMidpointIndex]));

                        bool needToSubdivideAgain = false;

                        T overallEstimate = h/MAKE_FLOAT(2.0) *(f[nonAdaptiveIndex] + f[nonAdaptiveEndIndex]);
                        T leftEstimate = h/MAKE_FLOAT(4.0) * (f[nonAdaptiveIndex] + f[nonAdaptiveMidpointIndex]);
                        T rightEstimate = h/MAKE_FLOAT(4.0) * (f[nonAdaptiveMidpointIndex] + f[nonAdaptiveEndIndex]);

                        I[nonAdaptiveMidpointIndex] = I[nonAdaptiveIndex] + leftEstimate;
                        I[nonAdaptiveEndIndex] = I[nonAdaptiveMidpointIndex] + rightEstimate;

                        // If the curDepth is n-1, then we have reached the end of the tree, so no subdivision
                        // is possible.
                        if( curDepth != (n-1) )
                        {
                            if( curDepth <= minimumTreeDepth )
                            {
                                needToSubdivideAgain = true;
                            }
                            else
                            {
                                if( fabsf(overallEstimate - (leftEstimate+rightEstimate))/MAKE_FLOAT(6.0) > localEpsilon &&
                                    localEpsilon > MAKE_FLOAT(1e-9) )
                                {
                                    needToSubdivideAgain = true;
                                }
                            }
                        }

                        // Algorithm to go to the next node in the graph.

                        if( needToSubdivideAgain )
                        {
                            // If we are subdividing, then the next iteration uses the same index,
                            // just with smaller interval.
                            adaptiveIndex = adaptiveIndex;
                            curDepth += 1;
                        }
                        else
                        {

                            // Update the adaptive values.
                            t[adaptiveIndex] = t[nonAdaptiveIndex];
                            t[adaptiveIndex + 1] = t[nonAdaptiveMidpointIndex];

                            f[adaptiveIndex] = f[nonAdaptiveIndex];
                            f[adaptiveIndex + 1] = f[nonAdaptiveMidpointIndex];

                            I[adaptiveIndex] = I[nonAdaptiveIndex];
                            I[adaptiveIndex + 1] = I[nonAdaptiveMidpointIndex];

                            // If we are not subdividing, then the value of the integral to this
                            // point has been calculated and we can move on.
                            adaptiveIndex += 2;
                            nonAdaptiveIndex += stride;

                            // Check possible strides, from the maximum to 2.
                            for(unsigned int j = n-1; j > 0; --j)
                            {
                                unsigned int checkStride = (0x01 << j);
                                if( nonAdaptiveIndex % checkStride == 0 )
                                {
                                    curDepth = n-j;
                                    break;
                                }
                            }

                            if( nonAdaptiveIndex == arraySize-1 )
                            {
                                t[adaptiveIndex] = t[nonAdaptiveEndIndex];
                                f[adaptiveIndex] = f[nonAdaptiveEndIndex];
                                I[adaptiveIndex] = I[nonAdaptiveEndIndex];
                                done = true;
                            }
                        }
                    }

                }

                template<typename DensityFuncType>
                RT_DEVICE T SampleInnerIntegral(T t_i, T sample, const DensityFuncType& densityFunc) const
                {
                    if( t_i < t[0] || 
                        t_i > t[adaptiveIndex] )
                    {
                        return MAKE_FLOAT(0.0);
                    }

                    if( t_i == t[0] ) return MAKE_FLOAT(0.0);
                    if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

                    const T* a = &(t[0]);
                    const T* b = &(t[adaptiveIndex]);
                    while(b-a > 1 )
                    {
                        const T* mid = (b-a)/MAKE_FLOAT(2.0) + a;
                        if( *mid == t_i ) 
                        {
                            return I[mid-a];
                        }
                        if( t_i < *mid )
                        {
                            b = mid;
                        }
                        else
                        {
                            a = mid;
                        }
                    }

                    T baseline = I[a-t];
                    T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc(sample));
                    return baseline+segment;
                }

                RT_DEVICE T OverallValue() const 
                {
                    return I[adaptiveIndex];
                }

                static const unsigned int arraySize = (0x01 << n) + 1;

                T t[arraySize];
                T f[arraySize];
                T I[arraySize];
            private:
                unsigned int adaptiveIndex;
        };


        template<typename T>
        RT_DEVICE T localFunction(const T& x)
        {
            return ((1.2 + 7.8*x)*x + 9.3)*x - 6.5;
        }

        // Uising local function, .05 seconds.
        // Using field sampling, 2 seconds
        template<typename T, unsigned int n>
        struct IterativeAdaptiveTrapezoidalIntegralFunctionRegistersOnly
        {
            public:
                template<typename TransferFunctionType, typename FieldFunctionType>
                RT_DEVICE void Integrate(T t0, T t1, const TransferFunctionType& transferFunction,
                    const FieldFunctionType& fieldFunction, const T epsilon)
                {
                    unsigned int curDepth = 1;

                    // The stride tells us both how wide the distance between samples in the
                    // array.
                    unsigned int nonAdaptiveIndex = 0;
                    adaptiveIndex = 0;

                    // The minimum number of subdivisions to perform before using
                    // adaptive quadrature.
                    const unsigned int minimumTreeDepth = 8;

                    T t_min = t0;
                    T t_max = t1;
                    T t_mid = t0 + (t1-t0)/MAKE_FLOAT(2.0);

                    T f_min = MAKE_FLOAT(0.0);
                    T f_mid = MAKE_FLOAT(0.0);
                    T f_max = MAKE_FLOAT(0.0);

                    T I_min = MAKE_FLOAT(0.0);
                    T I_mid = MAKE_FLOAT(0.0);
                    T I_max = MAKE_FLOAT(0.0);

                    //unsigned int index[] = {0, (arraySize-1)/2, arraySize-1};
                    ElVisFloat temp[] = {t0, (t1-t0)/MAKE_FLOAT(2.0) + t0, t1};
                    for(unsigned int i = 0; i < 3; ++i)
                    {
                        t_max = temp[i];
                        f_max = localFunction(temp[i]);//transferFunction(fieldFunction(temp[i]));
                    }

                    T range = t1-t0;

                    bool done = false;
                    // Sample.  Incrementing by 1 is a depth-first traversal.
                    while(!done)
                    {
                        unsigned int stride = (0x01 << (n-curDepth));

                        // Sample the midpoint of this interval.
                        //unsigned int nonAdaptiveMidpointIndex = nonAdaptiveIndex + stride/2;
                        //unsigned int nonAdaptiveEndIndex = nonAdaptiveIndex + stride;
                        T h = t_max - t_min;
                        T localEpsilon = epsilon * (h/range);
                        t_mid = t_min + h/MAKE_FLOAT(2.0);
                        f_mid = transferFunction(localFunction(t_mid));

                        bool needToSubdivideAgain = false;

                        T overallEstimate = h/MAKE_FLOAT(2.0) *(f_min + f_max);
                        T leftEstimate = h/MAKE_FLOAT(4.0) * (f_min + f_mid);
                        T rightEstimate = h/MAKE_FLOAT(4.0) * (f_mid + f_max);

                        I_mid = I_min + leftEstimate;
                        I_max = I_mid + rightEstimate;

                        // If the curDepth is n-1, then we have reached the end of the tree, so no subdivision
                        // is possible.
                        if( curDepth != (n-1) )
                        {
                            if( curDepth <= minimumTreeDepth )
                            {
                                needToSubdivideAgain = true;
                            }
                            else
                            {
                                if( fabsf(overallEstimate - (leftEstimate+rightEstimate))/MAKE_FLOAT(6.0) > localEpsilon &&
                                    localEpsilon > MAKE_FLOAT(1e-9) )
                                {
                                    needToSubdivideAgain = true;
                                }
                            }
                        }

                        // Algorithm to go to the next node in the graph.

                        if( needToSubdivideAgain )
                        {
                            // If we are subdividing, then the next iteration uses the same index,
                            // just with smaller interval.
                            adaptiveIndex = adaptiveIndex;
                            curDepth += 1;
                        }
                        else
                        {

                            // Update the adaptive values.
                            //t[adaptiveIndex] = t[nonAdaptiveIndex];
                            //t[adaptiveIndex + 1] = t[nonAdaptiveMidpointIndex];

                            //f[adaptiveIndex] = f[nonAdaptiveIndex];
                            //f[adaptiveIndex + 1] = f[nonAdaptiveMidpointIndex];

                            //I[adaptiveIndex] = I[nonAdaptiveIndex];
                            //I[adaptiveIndex + 1] = I[nonAdaptiveMidpointIndex];

                            // If we are not subdividing, then the value of the integral to this
                            // point has been calculated and we can move on.
                            adaptiveIndex += 2;
                            nonAdaptiveIndex += stride;

                            // Check possible strides, from the maximum to 2.
                            for(unsigned int j = n-1; j > 0; --j)
                            {
                                unsigned int checkStride = (0x01 << j);
                                if( nonAdaptiveIndex % checkStride == 0 )
                                {
                                    curDepth = n-j;
                                    break;
                                }
                            }

                            if( nonAdaptiveIndex == arraySize-1 )
                            {
                                //t[adaptiveIndex] = t[nonAdaptiveEndIndex];
                                //f[adaptiveIndex] = f[nonAdaptiveEndIndex];
                                //I[adaptiveIndex] = I[nonAdaptiveEndIndex];
                                done = true;
                            }
                        }
                    }

                }

//                template<typename DensityFuncType>
//                RT_DEVICE T SampleInnerIntegral(T t_i, T sample, const DensityFuncType& densityFunc) const
//                {
//                    if( t_i < t[0] ||
//                        t_i > t[adaptiveIndex] )
//                    {
//                        return MAKE_FLOAT(0.0);
//                    }

//                    if( t_i == t[0] ) return MAKE_FLOAT(0.0);
//                    if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

//                    const T* a = &(t[0]);
//                    const T* b = &(t[adaptiveIndex]);
//                    while(b-a > 1 )
//                    {
//                        const T* mid = (b-a)/MAKE_FLOAT(2.0) + a;
//                        if( *mid == t_i )
//                        {
//                            return I[mid-a];
//                        }
//                        if( t_i < *mid )
//                        {
//                            b = mid;
//                        }
//                        else
//                        {
//                            a = mid;
//                        }
//                    }

//                    T baseline = I[a-t];
//                    T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc(sample));
//                    return baseline+segment;
//                }

                RT_DEVICE T OverallValue() const
                {
                    return MAKE_FLOAT(.5);
                }

                static const unsigned int arraySize = (0x01 << n) + 1;

            private:
                unsigned int adaptiveIndex;
        };

        /// n - An upper bound on the number of recursive calls that can be made.  The higher n, the
        /// more memory this object will consume, the longer it will take to computer, and the
        /// more accurate the result.
        template<typename T, unsigned int n>
        struct IterativeAdaptiveTrapezoidalIntegralFunctionInterleaved
        {
            public:
                template<typename TransferFunctionType, typename FieldFunctionType>
                RT_DEVICE void Integrate(T t0, T t1, const TransferFunctionType& transferFunction,
                    const FieldFunctionType& fieldFunction, const T epsilon)
                {
                    unsigned int curDepth = 1;

                    // The stride tells us both how wide the distance between samples in the
                    // array.
                    unsigned int nonAdaptiveIndex = 0;
                    adaptiveIndex = 0;

                    // The minimum number of subdivisions to perform before using
                    // adaptive quadrature.
                    const unsigned int minimumTreeDepth = 8;

                    unsigned int index[] = {0, (arraySize-1)/2, arraySize-1};
                    ElVisFloat temp[] = {t0, (t1-t0)/MAKE_FLOAT(2.0) + t0, t1};
                    for(unsigned int i = 0; i < 3; ++i)
                    {
                        data[index[i]*4] = temp[i];
                        data[index[i]*4+1] = transferFunction(fieldFunction(temp[i]));
                    }

                    // Do the initial sampling before starting the loop.
//                    t[0] = t0;
//                    f[0] = transferFunction(fieldFunction(t[0]));

//                    t[arraySize-1] = t1;
//                    f[arraySize-1] = transferFunction(fieldFunction(t[arraySize-1]));

                    T range = t1-t0;
//                    t[(arraySize-1)/2] = (t1-t0)/2.0 + t0;
//                    f[(arraySize-1)/2] = transferFunction(fieldFunction(t[(arraySize-1)/2]));

                    data[0*4+2] = MAKE_FLOAT(0.0);

                    bool done = false;
                    // Sample.  Incrementing by 1 is a depth-first traversal.
                    while(!done)
                    {
                        unsigned int stride = (0x01 << (n-curDepth));

                        // Sample the midpoint of this interval.
                        unsigned int nonAdaptiveMidpointIndex = nonAdaptiveIndex + stride/2;
                        unsigned int nonAdaptiveEndIndex = nonAdaptiveIndex + stride;
                        T h = data[(nonAdaptiveIndex+stride)*4] - data[4*nonAdaptiveIndex];
                        T localEpsilon = epsilon * (h/range);
                        data[nonAdaptiveMidpointIndex*4] = data[nonAdaptiveIndex*4] + h/MAKE_FLOAT(2.0);
                        data[nonAdaptiveMidpointIndex*4+1] = transferFunction(fieldFunction(data[nonAdaptiveMidpointIndex*4]));

                        bool needToSubdivideAgain = false;

                        T overallEstimate = h/MAKE_FLOAT(2.0) *(data[nonAdaptiveIndex*4+1] + data[nonAdaptiveEndIndex*4+1]);
                        T leftEstimate = h/MAKE_FLOAT(4.0) * (data[nonAdaptiveIndex*4+1] + data[nonAdaptiveMidpointIndex*4+1]);
                        T rightEstimate = h/MAKE_FLOAT(4.0) * (data[nonAdaptiveMidpointIndex*4+1] + data[nonAdaptiveEndIndex*4+1]);

                        data[nonAdaptiveMidpointIndex*4+2] = data[nonAdaptiveIndex*4+2] + leftEstimate;
                        data[nonAdaptiveEndIndex*4+2] = data[nonAdaptiveMidpointIndex*4+2] + rightEstimate;

                        // If the curDepth is n-1, then we have reached the end of the tree, so no subdivision
                        // is possible.
                        if( curDepth != (n-1) )
                        {
                            if( curDepth <= minimumTreeDepth )
                            {
                                needToSubdivideAgain = true;
                            }
                            else
                            {
                                if( fabsf(overallEstimate - (leftEstimate+rightEstimate))/MAKE_FLOAT(6.0) > localEpsilon &&
                                    localEpsilon > MAKE_FLOAT(1e-9) )
                                {
                                    needToSubdivideAgain = true;
                                }
                            }
                        }

                        // Algorithm to go to the next node in the graph.

                        if( needToSubdivideAgain )
                        {
                            // If we are subdividing, then the next iteration uses the same index,
                            // just with smaller interval.
                            adaptiveIndex = adaptiveIndex;
                            curDepth += 1;
                        }
                        else
                        {

                            // Update the adaptive values.
                            data[adaptiveIndex*4] = data[nonAdaptiveIndex*4];
                            data[(adaptiveIndex + 1)*4] = data[nonAdaptiveMidpointIndex*4];

                            data[adaptiveIndex*4+1] = data[nonAdaptiveIndex*4+1];
                            data[(adaptiveIndex + 1)*4+1] = data[nonAdaptiveMidpointIndex*4+1];

                            data[adaptiveIndex*4+2] = data[nonAdaptiveIndex*4+2];
                            data[(adaptiveIndex + 1)*4+2] = data[nonAdaptiveMidpointIndex*4+2];

                            // If we are not subdividing, then the value of the integral to this
                            // point has been calculated and we can move on.
                            adaptiveIndex += 2;
                            nonAdaptiveIndex += stride;

                            // Check possible strides, from the maximum to 2.
                            for(unsigned int j = n-1; j > 0; --j)
                            {
                                unsigned int checkStride = (0x01 << j);
                                if( nonAdaptiveIndex % checkStride == 0 )
                                {
                                    curDepth = n-j;
                                    break;
                                }
                            }

                            if( nonAdaptiveIndex == arraySize-1 )
                            {
                                data[adaptiveIndex*4] = data[nonAdaptiveEndIndex*4];
                                data[adaptiveIndex*4+1] = data[nonAdaptiveEndIndex*4+1];
                                data[adaptiveIndex*4+2] = data[nonAdaptiveEndIndex*4+2];
                                done = true;
                            }
                        }
                    }

                }

//                template<typename DensityFuncType>
//                RT_DEVICE T SampleInnerIntegral(T t_i, T sample, const DensityFuncType& densityFunc) const
//                {
//                    if( t_i < t[0] ||
//                        t_i > t[adaptiveIndex] )
//                    {
//                        return MAKE_FLOAT(0.0);
//                    }

//                    if( t_i == t[0] ) return MAKE_FLOAT(0.0);
//                    if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

//                    const T* a = &(t[0]);
//                    const T* b = &(t[adaptiveIndex]);
//                    while(b-a > 1 )
//                    {
//                        const T* mid = (b-a)/MAKE_FLOAT(2.0) + a;
//                        if( *mid == t_i )
//                        {
//                            return I[mid-a];
//                        }
//                        if( t_i < *mid )
//                        {
//                            b = mid;
//                        }
//                        else
//                        {
//                            a = mid;
//                        }
//                    }

//                    T baseline = I[a-t];
//                    T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc(sample));
//                    return baseline+segment;
//                }

                RT_DEVICE T OverallValue() const
                {
                    return data[adaptiveIndex*3+2];
                }

                static const unsigned int arraySize = (0x01 << n) + 1;

                T data[arraySize*4];
            private:
                unsigned int adaptiveIndex;
        };

        /// n - An upper bound on the number of recursive calls that can be made.  The higher n, the
        /// more memory this object will consume, the longer it will take to computer, and the
        /// more accurate the result.
        template<typename T, unsigned int n>
        struct IterativeAdaptiveTrapezoidalIntegralFunctionStackVersion
        {
            public:
                struct StackPoint
                {
                    template<typename DensityTransferFunc,
                        typename FieldFunc>
                    RT_DEVICE
                    void Evaluate(const DensityTransferFunc& densityFunc,
                        const FieldFunc& fieldFunc)
                    {
                        T s = fieldFunc(TVal);
                        F = densityFunc(s);
                    }

                    RT_DEVICE void Reset()
                    {
                        TVal = MAKE_FLOAT(1e30);
                    }

                    RT_DEVICE bool IsUninitialized() const
                    {
                        return TVal == MAKE_FLOAT(1e30);
                    }

                    RT_DEVICE StackPoint& operator=(const StackPoint& rhs)
                    {
                        TVal = rhs.TVal;
                        F = rhs.F;
                        return *this;
                    }

                    T TVal;
                    T F;
                };




                struct StackEntry
                {
                    RT_DEVICE void CalculateMidpointT()
                    {
                        Mid().TVal = Left().TVal + (Right().TVal - Left().TVal)/2.0;
                    }

                    RT_DEVICE void SetT(const T& t0, const T& t1)
                    {
                        Left().TVal = t0;
                        Right().TVal = t1;
                        CalculateMidpointT();
                    }

                    RT_DEVICE T GetH() const
                    {
                        return Right().TVal - Left().TVal;
                    }

                    template<typename DensityTransferFunc, typename FieldFunc>
                    RT_DEVICE void EvaluateAll(const DensityTransferFunc& densityFunc,
                        const FieldFunc& fieldFunc)
                    {
                        for(unsigned int i = 0; i < 3; ++i)
                        {
                            points[i].Evaluate(densityFunc, fieldFunc);
                        }
                    }

                    RT_DEVICE StackPoint& Left()  { return points[0]; }
                    RT_DEVICE StackPoint& Mid()  { return points[1]; }
                    RT_DEVICE StackPoint& Right() { return points[2]; }

                    RT_DEVICE const StackPoint& Left() const  { return points[0]; }
                    RT_DEVICE const StackPoint& Mid() const  { return points[1]; }
                    RT_DEVICE const StackPoint& Right() const { return points[2]; }

                    StackPoint points[3];
                };


                template<typename TransferFunctionType, typename FieldFunctionType>
                RT_DEVICE void Integrate(T t0, T t1, const TransferFunctionType& transferFunction,
                    const FieldFunctionType& fieldFunction, const T epsilon)
                {
                    const unsigned int maxRecursion = n;

                    StackEntry stack[maxRecursion];


                    stack[0].SetT(t0, t1);
                    stack[0].EvaluateAll(transferFunction, fieldFunction);
//                    stack[0].Left.Evaluate(transferFunction, fieldFunction);
//                    stack[0].Mid.Evaluate(transferFunction, fieldFunction);
//                    stack[0].Right.Evaluate(transferFunction, fieldFunction);

                    stack[1].Left() = stack[0].Left();
                    stack[1].Mid().Reset();
                    stack[1].Right() = stack[0].Mid();

                    unsigned int minimumDepth = n-2;

                    int i = 1;
                    t[0] = t0;
                    f[0] = stack[0].Left().F;
                    I[0] = 0.0;
                    adaptiveIndex = 0;
                    while( i > 0 )
                    {
                        if( stack[i].Mid().IsUninitialized() )
                        {
                            bool needToSubdivide = false;

                            stack[i].CalculateMidpointT();
                            stack[i].Mid().Evaluate(transferFunction, fieldFunction);

                            if( i < minimumDepth )
                            {
                                needToSubdivide = true;
                            }
                            else
                            {
                                T I0 = stack[i].GetH()/MAKE_FLOAT(2.0) * (stack[i].Left().F + stack[i].Right().F);
                                T I1 = stack[i].GetH()/MAKE_FLOAT(4.0) * (stack[i].Left().F + 2.0*stack[i].Mid().F + stack[i].Right().F);
                                T diff = fabs(I0-I1)/MAKE_FLOAT(6.0);

                                T localEpsilon = epsilon * (stack[i].GetH()/stack[0].GetH());
                                if( diff > localEpsilon && i < maxRecursion-1 )
                                {
                                    needToSubdivide = true;
                                }
                            }

                            if( needToSubdivide )
                            {
                                stack[i+1].Left() = stack[i].Left();
                                stack[i+1].Mid().Reset();
                                stack[i+1].Right() = stack[i].Mid();
                                i = i + 1;
                            }
                            else
                            {
                                T prevValue = I[adaptiveIndex];
                                T h = stack[i].GetH()/MAKE_FLOAT(4.0);
                                T mid_f = stack[i].Mid().F;
                                T right_f = stack[i].Right().F;

                                t[adaptiveIndex+1] = stack[i].Mid().TVal;
                                t[adaptiveIndex+2] = stack[i].Right().TVal;
                                f[adaptiveIndex+1] = mid_f;
                                f[adaptiveIndex+2] = right_f;

                                T leftContribution = h * (stack[i].Left().F + mid_f);
                                T rightContribution = h * (mid_f + right_f);

                                I[adaptiveIndex+1] = prevValue + leftContribution;
                                I[adaptiveIndex+2] = prevValue + leftContribution+rightContribution;
                                adaptiveIndex += 2;
                            }
                        }
                        else
                        {
                            if( stack[i].Right().TVal == stack[i-1].Mid().TVal )
                            {
                                // We just finished traversing the left side, now go to
                                // the right.
                                stack[i].Left() = stack[i-1].Mid();
                                stack[i].Mid().Reset();
                                stack[i].Right() = stack[i-1].Right();
                            }
                            else
                            {
                                // We finished this branch.  Remove it and go up to
                                // the next one.
                                i = i-1;
                            }
                        }
                    }

                }

                template<typename DensityFuncType>
                RT_DEVICE T SampleInnerIntegral(T t_i, T sample, const DensityFuncType& densityFunc) const
                {
                    if( t_i < t[0] ||
                        t_i > t[adaptiveIndex] )
                    {
                        return MAKE_FLOAT(0.0);
                    }

                    if( t_i == t[0] ) return MAKE_FLOAT(0.0);
                    if( t_i == t[adaptiveIndex] ) return I[adaptiveIndex];

                    const T* a = &(t[0]);
                    const T* b = &(t[adaptiveIndex]);
                    while(b-a > 1 )
                    {
                        const T* mid = (b-a)/2 + a;
                        if( *mid == t_i )
                        {
                            return I[mid-a];
                        }
                        if( t_i < *mid )
                        {
                            b = mid;
                        }
                        else
                        {
                            a = mid;
                        }
                    }

                    T baseline = I[a-t];
                    T segment = (t_i-*a)/MAKE_FLOAT(2.0) * ( f[a-t] + densityFunc(sample));
                    return baseline+segment;
                }

                RT_DEVICE T OverallValue() const
                {
                    return I[adaptiveIndex];
                }

                static const unsigned int arraySize = (0x01 << n) + 1;

                T t[arraySize];
                T f[arraySize];
                T I[arraySize];
            private:
                unsigned int adaptiveIndex;
        };
    }
}

#endif
