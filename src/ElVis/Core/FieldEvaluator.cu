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

#ifndef ELVIS_CORE_FIELD_EVALUATOR_CU
#define ELVIS_CORE_FIELD_EVALUATOR_CU

#include <ElVis/Core/Cuda.h>

struct FieldEvaluator
{
    ELVIS_DEVICE FieldEvaluator() :
      Origin(),
      Direction(),
      ElementId(0),
      ElementType(0),
      FieldId(0),
      sampleCount(0)
    {
    }

    ELVIS_DEVICE ElVisFloat operator()(const ElVisFloat& t) const
    {
        ElVisFloat3 p = Origin + t*Direction;
#ifdef ELVIS_OPTIX_MODULE
        ElVisFloat s = EvaluateFieldOptiX(ElementId, ElementType, FieldId, p);
#else
        ElVisFloat s = EvaluateFieldCuda(ElementId, ElementType, FieldId, p);
#endif

        if( sampleCount )
        {
            atomicAdd(sampleCount, 1);
        }
        return s;
    }

    ELVIS_DEVICE ElVis::Interval<ElVisFloat> EstimateRange(const ElVisFloat& t0, const ElVisFloat& t1) const
    {
        //ElVisFloat3 p0 = Origin + t0*Direction;
        //ElVisFloat3 p1 = Origin + t1*Direction;
        ElVis::Interval<ElVisFloat> result;
        //::EstimateRangeOptiX(ElementId, ElementType, FieldId, p0, p1, result);
        return result;
    }

    ELVIS_DEVICE void AdjustSampleCount(int value)
    {
        if( sampleCount )
        {
            atomicAdd(sampleCount, value);
        }
    }

    ElVisFloat3 Origin;
    ElVisFloat3 Direction;
    unsigned int ElementId;
    unsigned int ElementType;
    int FieldId;
    int* sampleCount;
};

#endif //ELVIS_CORE_FIELD_EVALUATOR_CU
