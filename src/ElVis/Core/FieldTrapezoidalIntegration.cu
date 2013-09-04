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

#ifndef ELVIS_CORE_FIELD_TRAPEZOIDAL_INTEGRATION_CU
#define ELVIS_CORE_FIELD_TRAPEZOIDAL_INTEGRATION_CU

#include <ElVis/Core/Float.h>

namespace ElVis
{
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
}

#endif
