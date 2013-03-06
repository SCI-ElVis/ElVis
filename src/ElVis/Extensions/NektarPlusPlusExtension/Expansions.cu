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

#ifndef ELVIS_EXPANSIONS_CU
#define ELVIS_EXPANSIONS_CU

#include <ElVis/Core/jacobi.cu>
#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Cuda.h>

ELVIS_DEVICE ElVisFloat IntegerPow(const ElVisFloat& base, unsigned int power)
{
    if( power == 0 ) return MAKE_FLOAT(1.0);
    if( power == 1 ) return base;

    ElVisFloat result = base;
    for(unsigned int i = 1; i < power; ++i)
    {
        result = result*base;
    }
    return result;
}

template<typename T>
ELVIS_DEVICE T ModifiedA(unsigned int i, const T& x)
{
    if( i == 0 )
    {
        return (MAKE_FLOAT(1.0)-x)/MAKE_FLOAT(2.0);
    }
    else if( i == 1 )
    {
        return (MAKE_FLOAT(1.0)+x)/MAKE_FLOAT(2.0);
    }
    else
    {
        return (MAKE_FLOAT(1.0)-x)/MAKE_FLOAT(2.0) * 
            (MAKE_FLOAT(1.0) + x)/MAKE_FLOAT(2.0) * 
            ElVis::OrthoPoly::P(i-2, 1, 1, x);
    }
}

template<typename T>
ELVIS_DEVICE T ModifiedAPrime(unsigned int i, const T& x)
{
    if( i == 0 )
    {
        return MAKE_FLOAT(-.5);
    }
    else if( i == 1 )
    {
        return MAKE_FLOAT(.5);
    }
    else
    {
        ElVisFloat poly = ElVis::OrthoPoly::P(i-2, 1, 1, x);
        ElVisFloat dpoly = ElVis::OrthoPoly::dP(i-2, 1, 1, x);

        return MAKE_FLOAT(.25)*(1-x)*poly - MAKE_FLOAT(.25) * (1+x)*poly + 
            MAKE_FLOAT(.25) * (1-x)*(1+x)*dpoly;
    }
}

template<typename T>
ELVIS_DEVICE void ModifiedA(int i, const T& x, T* out)
{
    out[0] = (MAKE_FLOAT(1.0)-x)*MAKE_FLOAT(.5);
    if( i >= 1 )
    {
        out[1] = (MAKE_FLOAT(1.0)+x)*MAKE_FLOAT(.5);
    }
    if( i >= 2 )
    {
        ElVis::OrthoPoly::P(i-2, 1, 1, x, out+2);
        for(int j = 2; j <= i; ++j)
        {
            out[j] *= (MAKE_FLOAT(1.0)-x)*MAKE_FLOAT(.5) *
            (MAKE_FLOAT(1.0) + x)*MAKE_FLOAT(.5);
        }
    }
}

template<typename T>
ELVIS_DEVICE T ModifiedB(unsigned int i, unsigned int j, const T& x)
{
    if( i == 0 )
    {
        return ModifiedA(j, x);
    }
    else if( j == 0 )
    {
        return IntegerPow((1.0-x)/2.0, i);
    }
    else
    {
        ElVisFloat result = 1.0;
        result = IntegerPow((1.0-x)/2.0, i);
        result *= (1.0+x)/2.0;
        result *= ElVis::OrthoPoly::P(j-1, 2*i-1, 1, x);
        return result;
    }
}

        
template<typename T>
ELVIS_DEVICE T ModifiedBPrime(unsigned int i, unsigned int j, const T& x)
{
    if( i == 0 )
    {
        return ModifiedAPrime(j, x);
    }
    else if( j == 0 )
    {
        ElVisFloat result = MAKE_FLOAT(1.0)/IntegerPow(MAKE_FLOAT(-2.0), i);
        //ELVIS_PRINTF("ModifiedBPrime: i %d j %d first result %f\n", i, j, result);
        result *= i;
        //ELVIS_PRINTF("ModifiedBPrime: i %d j %d multiply i %f\n", i, j, result);
        result *= IntegerPow((MAKE_FLOAT(1.0)-x), i-1);
        //ELVIS_PRINTF("ModifiedBPrime: i %d j %d result %f\n", i, j, result);
        return result;
    }
    else
    {
        ElVisFloat scale = MAKE_FLOAT(1.0)/IntegerPow(MAKE_FLOAT(2.0), i+1);
        ElVisFloat poly = ElVis::OrthoPoly::P(j-1, 2*i-1, 1, x);
        ElVisFloat dpoly = ElVis::OrthoPoly::dP(j-1, 2*i-1, 1, x);
        ElVisFloat result = IntegerPow(MAKE_FLOAT(1.0)-x, i) * poly -
            i*IntegerPow(MAKE_FLOAT(1.0)-x, i-1)*(1+x)*poly +
            IntegerPow(MAKE_FLOAT(1.0)-x, i)*(1+x)*dpoly;
        result *= scale;
        //ELVIS_PRINTF("ModifiedBPrime: i %d j %d result %f\n", i, j, result);
        return result;
    }
}

#endif
