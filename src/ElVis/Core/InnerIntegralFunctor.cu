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

#ifndef ELVIS_CORE_INNER_INTEGRAL_FUNCTOR_CU
#define ELVIS_CORE_INNER_INTEGRAL_FUNCTOR_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/TransferFunction.h>

namespace ElVis
{
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
}

#endif
