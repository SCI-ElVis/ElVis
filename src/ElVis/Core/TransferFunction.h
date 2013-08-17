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

#ifndef ELVIS_CORE_TRANSFER_FUNCTION_H
#define ELVIS_CORE_TRANSFER_FUNCTION_H

#include <ElVis/Core/Cuda.h>
#include <ElVis/Core/Float.h>
#include <ElVis/Core/Interval.hpp>
#include <ElVis/Core/ElVisDeclspec.h>

namespace ElVis
{
    enum TransferFunctionChannel
    {
        eDensity,
        eRed, 
        eGreen,
        eBlue
    };

    class TransferFunction
    {
        public:
            // This class is meant to be used in Cuda and C++, so we need empty
            // constructors.  Clients should call Initialize() before use.
            ELVIS_DEVICE TransferFunction() {}

            ELVIS_DEVICE void Initialize()
            {
                m_densityBreakpoints = 0;
                m_redBreakpoints = 0;
                m_greenBreakpoints = 0;
                m_blueBreakpoints = 0;

                m_densityValues = 0;
                m_redValues = 0;
                m_greenValues = 0;
                m_blueValues = 0;

                m_numDensityBreakpoints = 0;
                m_numRedBreakpoints = 0;
                m_numGreenBreakpoints = 0;
                m_numBlueBreakpoints = 0;
            }

            ELVIS_DEVICE void GetBreakpointsForChannel(TransferFunctionChannel channel, ElVisFloat*& begin, ElVisFloat*& end, int& numBreakpoints) const
            {
                switch(channel)
                {
                    case eDensity:
                        begin = m_densityBreakpoints;
                        end = m_densityBreakpoints + (m_numDensityBreakpoints-1);
                        numBreakpoints = m_numDensityBreakpoints;
                        break;

                    case eRed:
                        begin = m_redBreakpoints;
                        end = m_redBreakpoints + (m_numRedBreakpoints-1);
                        numBreakpoints = m_numRedBreakpoints;
                        break;

                    case eGreen:
                        begin = m_greenBreakpoints;
                        end = m_greenBreakpoints + (m_numGreenBreakpoints-1);
                        numBreakpoints = m_numGreenBreakpoints;
                        break;

                    case eBlue:
                        begin = m_blueBreakpoints;
                        end = m_blueBreakpoints + (m_numBlueBreakpoints-1);
                        numBreakpoints = m_numBlueBreakpoints;
                        break;

                    default:
                        begin = m_densityBreakpoints;
                        end = m_densityBreakpoints + (m_numDensityBreakpoints-1);
                        numBreakpoints = m_numDensityBreakpoints;
                        break;
                }
            }

            ELVIS_DEVICE void GetValuesForChannel(TransferFunctionChannel channel, ElVisFloat*& begin, ElVisFloat*& end, int& count) const
            {
                switch(channel)
                {
                    case eDensity:
                        begin = m_densityValues;
                        end = m_densityValues + (m_numDensityBreakpoints-1);
                        count = m_numDensityBreakpoints;
                        break;

                    case eRed:
                        begin = m_redValues;
                        end = m_redValues + (m_numRedBreakpoints-1);
                        count = m_numRedBreakpoints;
                        break;

                    case eGreen:
                        begin = m_greenValues;
                        end = m_greenValues + (m_numGreenBreakpoints-1);
                        count = m_numGreenBreakpoints;
                        break;

                    case eBlue:
                        begin = m_blueValues;
                        end = m_blueValues + (m_numBlueBreakpoints-1);
                        count = m_numBlueBreakpoints;
                        break;

                    default:
                        begin = m_densityValues;
                        end = m_densityValues + (m_numDensityBreakpoints-1);
                        count = m_numDensityBreakpoints;
                        break;
                }
            }

            ELVIS_DEVICE bool ColorContainsAtLeastOneBreakpoint(const Interval<ElVisFloat>& range) const
            {
                return RangeContainsAtLeastOneBreakpoint(eGreen, range) ||
                        RangeContainsAtLeastOneBreakpoint(eRed, range) ||
                        RangeContainsAtLeastOneBreakpoint(eBlue, range);
            }



            ELVIS_DEVICE bool RangeContainsAtLeastOneBreakpoint(TransferFunctionChannel channel, const Interval<ElVisFloat>& range) const
            {
                ElVisFloat* breakpointStart = 0;
                ElVisFloat* breakpointEnd = 0;
                int numBreakpoints = 0;
                GetBreakpointsForChannel(channel, breakpointStart, breakpointEnd, numBreakpoints);

                for(int i = 0; i < numBreakpoints; ++i)
                {
                    if( range.Contains(breakpointStart[i]) )
                    {
                        return true;
                    }
                }

                return false;
            }

            ELVIS_DEVICE bool IntersectsRange(TransferFunctionChannel channel, const Interval<ElVisFloat>& range) const
            { 
                ElVisFloat* breakpointStart = 0;
                ElVisFloat* breakpointEnd = 0;
                int numBreakpoints = 0;
                GetBreakpointsForChannel(channel, breakpointStart, breakpointEnd, numBreakpoints);

                return (*breakpointEnd) >= range.GetLow() &&
                    (*breakpointStart) <= range.GetHigh();
            }

//            ELVIS_DEVICE Interval<ElVisFloat> Sample(TransferFunctionChannel channel, const Interval<ElVisFloat>& range) const
//            {
//                Interval<ElVisFloat> result;

//                ElVisFloat* breakpointStart = 0;
//                ElVisFloat* breakpointEnd = 0;
//                int numBreakpoints = 0;
//                GetBreakpointsForChannel(channel, breakpointStart, breakpointEnd, numBreakpoints);

//                // Find the location of the breakpoint.
//                int index0 = 0;
//                int index1 = 0;
//                for(int i = 0; i < numBreakpoints-1; ++i)
//                {
//                    bool test0 = range.GetLow() >= breakpointStart[i];
//                    test0 &= (range.GetLow() < breakpointStart[i+1]);

//                    if( test0 ) index0 = i+1;

//                    bool test1 = range.GetHigh() >= breakpointStart[i];
//                    test1 &= (range.GetHigh() < breakpointStart[i+1]);

//                    if( test1 ) index1 = i+1;
//                }
//                if( range.GetLow() > breakpointStart[numBreakpoints-1] )
//                {
//                    index0 = numBreakpoints;
//                }
//                if( range.GetHigh() > breakpointStart[numBreakpoints-1] )
//                {
//                    index1 = numBreakpoints;
//                }

//                ElVisFloat* valueBegin = 0;
//                ElVisFloat* valueEnd = 0;
//                int count = 0;
//                GetValuesForChannel(channel, valueBegin, valueEnd, count);

//                if( index0 == 0 )
//                {
//                    result.SetLow(valueBegin[0]);
//                }
//                else if( index0 == numBreakpoints )
//                {
//                    result.SetLow(valueBegin[index0-1]);
//                }
//                else
//                {
//                    ElVisFloat v0 = valueBegin[index0-1];
//                    ElVisFloat v1 = valueBegin[index0];

//                    ElVisFloat s0 = breakpointStart[index0-1];
//                    ElVisFloat s1 = breakpointStart[index0];

//                    if( v0 == v1 )
//                    {
//                        result.SetLow(v0);
//                    }
//                    else
//                    {
//                        ElVisFloat scale = MAKE_FLOAT(1.0)/(s1-s0);
//                        result.SetLow(scale*v1*(range.GetLow()-s0) + scale*v0*(s1-range.GetLow()));
//                    }
//                }


//                if( index1 == 0 )
//                {
//                    result.SetHigh(valueBegin[0]);
//                }
//                else if( index1 == numBreakpoints )
//                {
//                    result.SetHigh(valueBegin[index1-1]);
//                }
//                else
//                {
//                    ElVisFloat v0 = valueBegin[index1-1];
//                    ElVisFloat v1 = valueBegin[index1];

//                    ElVisFloat s0 = breakpointStart[index1-1];
//                    ElVisFloat s1 = breakpointStart[index1];

//                    if( v0 == v1 )
//                    {
//                        result.SetHigh(v0);
//                    }
//                    else
//                    {
//                        ElVisFloat scale = MAKE_FLOAT(1.0)/(s1-s0);
//                        result.SetHigh(scale*v1*(range.GetHigh()-s0) + scale*v0*(s1-range.GetHigh()));
//                    }
//                }


//                for(int i = index0; i < index1; ++i)
//                {
//                    if( valueBegin[i] < result.GetLow() )
//                    {
//                        result.SetLow(valueBegin[i]);
//                    }
//                    if( valueBegin[i] > result.GetHigh() )
//                    {
//                        result.SetHigh(valueBegin[i]);
//                    }
//                }
//                return result;
//            }

            // Not as optimal as the above version is supposed to be, but it appears it has a bug, and I
            // need something correct right now.
            ELVIS_DEVICE Interval<ElVisFloat> Sample(TransferFunctionChannel channel, const Interval<ElVisFloat>& range) const
            {
                Interval<ElVisFloat> result;
                ElVisFloat left = Sample(channel, range.GetLow());
                ElVisFloat right = Sample(channel, range.GetHigh());
                result.Set(fminf(left, right), fmaxf(left, right));

                ElVisFloat* breakpointStart = 0;
                ElVisFloat* breakpointEnd = 0;
                int numBreakpoints = 0;
                GetBreakpointsForChannel(channel, breakpointStart, breakpointEnd, numBreakpoints);

                ElVisFloat* valueBegin = 0;
                ElVisFloat* valueEnd = 0;
                int count = 0;
                GetValuesForChannel(channel, valueBegin, valueEnd, count);

                for(int i = 0; i < numBreakpoints; ++i)
                {
                    if( range.Contains(breakpointStart[i]) )
                    {
                        result.SetLow(fminf(result.GetLow(), valueBegin[i]));
                        result.SetHigh(fmaxf(result.GetHigh(), valueBegin[i]));
                    }
                }
                return result;
            }

            ELVIS_DEVICE ElVisFloat Sample(TransferFunctionChannel channel, const ElVisFloat& s) const
            {
                ElVisFloat* breakpointStart = 0;
                ElVisFloat* breakpointEnd = 0;
                int numBreakpoints = 0;
                GetBreakpointsForChannel(channel, breakpointStart, breakpointEnd, numBreakpoints);

                // Find the location of the breakpoint.
                int index = 0;
                for(int i = 0; i < numBreakpoints-1; ++i)
                {
                    bool test = s >= breakpointStart[i];
                    test &= (s < breakpointStart[i+1]);

                    if( test ) index = i+1;
                }
                if( s >= breakpointStart[numBreakpoints-1] )
                {
                    index = numBreakpoints;
                }

                ElVisFloat* valueBegin = 0;
                ElVisFloat* valueEnd = 0;
                int count = 0;
                GetValuesForChannel(channel, valueBegin, valueEnd, count);

                ElVisFloat result = MAKE_FLOAT(0.0);
                if( index == 0 )
                {
                    result = valueBegin[0];
                }
                else if( index == numBreakpoints )
                {
                    result = valueBegin[index-1];
                }
                else
                {
                    ElVisFloat v0 = valueBegin[index-1];
                    ElVisFloat v1 = valueBegin[index];

                    ElVisFloat s0 = breakpointStart[index-1];
                    ElVisFloat s1 = breakpointStart[index];

                    if( v0 == v1 )
                    {
                        result = v0;
                    }
                    else
                    {
                        ElVisFloat scale = MAKE_FLOAT(1.0)/(s1-s0);
                        result = scale*v1*(s-s0) + scale*v0*(s1-s);
                    }
                }

                return result;
            }

            ELVIS_DEVICE ElVisFloat3 SampleColor(const ElVisFloat& s) const
            {
                ElVisFloat3 result;
                result.x = Sample(eRed, s);
                result.y = Sample(eGreen, s);
                result.z = Sample(eBlue, s);


                return result;
            }

            ELVIS_DEVICE ElVisFloat GetMaxValue(TransferFunctionChannel channel) const
            {
                ElVisFloat* begin = 0;
                ElVisFloat* end = 0;
                int count = 0;
                GetValuesForChannel(channel, begin, end, count);
                ElVisFloat result = MAKE_FLOAT(0.0);

                for(int i = 0; i < count; ++i)
                {
                    result = fmaxf(begin[i], result);
                }
                return result;
            }

            ELVIS_DEVICE ElVisFloat GetMaxValue(TransferFunctionChannel channel, const ElVis::Interval<ElVisFloat>& range) const
            {
                ElVisFloat* valueBegin = 0;
                ElVisFloat* valueEnd = 0;
                int count = 0;
                GetValuesForChannel(channel, valueBegin, valueEnd, count);

                ElVisFloat* breakpointBegin = 0;
                ElVisFloat* breakpointEnd = 0;
                GetBreakpointsForChannel(channel, breakpointBegin, breakpointEnd, count);

                ElVisFloat result = MAKE_FLOAT(0.0);

                for(int i = 0; i < count; ++i)
                {
                    if( range.Contains(breakpointBegin[i]) )
                    {
                        result = fmaxf(valueBegin[i], result);
                    }
                }
                result = fmaxf(Sample(channel, range.GetLow()), result);
                result = fmaxf(Sample(channel, range.GetHigh()), result);
                return result;
            }

            ELVIS_DEVICE ElVisFloat*& DensityBreakpoints() { return m_densityBreakpoints; }
            ELVIS_DEVICE ElVisFloat*& RedBreakpoints() { return m_redBreakpoints; }
            ELVIS_DEVICE ElVisFloat*& GreenBreakpoints() { return m_greenBreakpoints; }
            ELVIS_DEVICE ElVisFloat*& BlueBreakpoints() { return m_blueBreakpoints; }

            ELVIS_DEVICE ElVisFloat*& DensityValues() { return m_densityValues; }
            ELVIS_DEVICE ElVisFloat*& RedValues() { return m_redValues; }
            ELVIS_DEVICE ElVisFloat*& GreenValues() { return m_greenValues; }
            ELVIS_DEVICE ElVisFloat*& BlueValues() { return m_blueValues; }

            ELVIS_DEVICE int& NumDensityBreakpoints() { return m_numDensityBreakpoints; }
            ELVIS_DEVICE int& NumRedBreakpoints() { return m_numRedBreakpoints; }
            ELVIS_DEVICE int& NumGreenBreakpoints() { return m_numGreenBreakpoints; }
            ELVIS_DEVICE int& NumBlueBreakpoints() { return m_numBlueBreakpoints; }

        private:
            TransferFunction(const TransferFunction&);
            TransferFunction& operator=(const TransferFunction&);

            ElVisFloat* m_densityBreakpoints;
            ElVisFloat* m_redBreakpoints;
            ElVisFloat* m_greenBreakpoints;
            ElVisFloat* m_blueBreakpoints;

            ElVisFloat* m_densityValues;
            ElVisFloat* m_redValues;
            ElVisFloat* m_greenValues;
            ElVisFloat* m_blueValues;

            int m_numDensityBreakpoints;
            int m_numRedBreakpoints;
            int m_numGreenBreakpoints;
            int m_numBlueBreakpoints;

    };
}

#endif
