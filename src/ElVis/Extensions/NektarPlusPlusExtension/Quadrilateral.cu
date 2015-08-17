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

#ifndef ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_QUADRILATERAL_CU
#define ELVIS_EXTENSIONS_NEKTAR_PLUS_PLUS_EXTENSION_OPTIX_QUADRILATERAL_CU

#include <ElVis/Extensions/NektarPlusPlusExtension/Expansions.cu>

__device__ __forceinline__ ElVisFloat EvaluateQuadAtReferencePoint(
    ElVisFloat *coeffs, uint2 *modes, const ElVisFloat2& p)
{
    int cnt = 0;
    ElVisFloat result = MAKE_FLOAT(0.0);

    for(unsigned int j = 0; j < modes->y; ++j)
    {
        ElVisFloat value_j = ModifiedA(j, p.y);
        for(unsigned int i = 0; i < modes->x; ++i)
        {
            result += coeffs[cnt++] * ModifiedA(i, p.x) * value_j;
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluateQuadGradientAtReferencePoint0(
    ElVisFloat *coeffs, uint2 *modes, const ElVisFloat2& p)
{
    int cnt = 0;
    ElVisFloat result = MAKE_FLOAT(0.0);

    for(unsigned int j = 0; j < modes->y; ++j)
    {
        ElVisFloat value_j = ModifiedA(j, p.y);
        for(unsigned int i = 0; i < modes->x; ++i)
        {
            result += coeffs[cnt++] * ModifiedAPrime(i, p.x) * value_j;
        }
    }

    return result;
}

__device__ __forceinline__ ElVisFloat EvaluateQuadGradientAtReferencePoint1(
    ElVisFloat *coeffs, uint2 *modes, const ElVisFloat2& p)
{
    int cnt = 0;
    ElVisFloat result = MAKE_FLOAT(0.0);

    for(unsigned int j = 0; j < modes->y; ++j)
    {
        ElVisFloat value_j = ModifiedAPrime(j, p.y);
        for(unsigned int i = 0; i < modes->x; ++i)
        {
            result += coeffs[cnt++] * ModifiedA(i, p.x) * value_j;
        }
    }

    return result;
}

#endif
