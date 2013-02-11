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

#ifndef ELVIS_JACOBI_CU
#define ELVIS_JACOBI_CU

#include <ElVis/Core/Float.cu>
#include <ElVis/Core/Jacobi.hpp>

//#include <Nektar/FiniteElementMath.h>
//
//#include <iostream>
//#include <math.h>

// Jacobi polynomials and derivatives.
// Reference the book.
// Note that this file contains mostly generated code and
// therefore care should be taken if modifications are needed.
//namespace Jacobi
//{
//    // Evaluates the nth Jacobi polynomial at point x.
//    template<typename DataType>
//    __device__ __forceinline__ DataType Legendre(int n, const DataType& x);
//
//    template<typename DataType>
//    __device__ __forceinline__ DataType P(int n, int a, int b, const DataType& x);
//
//    template<typename DataType>
//    __device__ __forceinline__ DataType dP(int n, int a, int b, const DataType& x);
//
//    template<typename DataType>
//    __device__ __forceinline__ DataType ddP(int n, int a, int b, const DataType& x);
//}
//
//namespace Jacobi
//{
//    template<typename DataType>
//    __device__ __forceinline__ DataType Legendre(int n, const DataType& x)
//    {
//        return P(n, 0, 0, x);
//    }
//
//    // Evaluates the nth Jacobi polynomial at point x.
//    template<typename DataType>
//    __device__ __forceinline__ DataType P(int n, int a, int b, const DataType& x)
//    {
//        // From spectral methods page 351.
//        if( n == 0 )
//        {
//            return DataType(1.0);
//        }
//
//        if( n == 1 )
//        {
//            return .5*((ElVisFloat)a-(ElVisFloat)b+((ElVisFloat)a+(ElVisFloat)b+MAKE_FLOAT(2.0))*x);
//        }
//
//        DataType result = DataType(MAKE_FLOAT(0.0));
//        ElVisFloat   apb = a + b;
//
//        DataType polyn2 = DataType(MAKE_FLOAT(1.0));
//        DataType polyn1 = MAKE_FLOAT(0.5)*( static_cast<ElVisFloat>(a - b) + (a + b + MAKE_FLOAT(2.0))*x);
//        ElVisFloat alpha = a;
//        ElVisFloat beta = b;
//        for(int k = 2; k <= n; ++k)
//        {
//            ElVisFloat a1 = MAKE_FLOAT(2.0)*k*(k + apb)*(MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(2.0));
//            ElVisFloat a2 = (MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(1.0))*(alpha*alpha - beta*beta);
//            ElVisFloat a3 = (MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(2.0))*(MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(1.0))*(MAKE_FLOAT(2.0)*k + apb);
//            ElVisFloat a4 = MAKE_FLOAT(2.0)*(k + alpha - MAKE_FLOAT(1.0))*(k + beta - MAKE_FLOAT(1.0))*(MAKE_FLOAT(2.0)*k + apb);
//
//            a2 /= a1;
//            a3 /= a1;
//            a4 /= a1;
//
//
//            result = (a2 + a3*x)*polyn1 - a4*polyn2;
//
//            polyn2 = polyn1;
//            polyn1 = result;
//        }
//
//
//        return result;
//    }
//
//    template<typename DataType>
//    __device__ __forceinline__ void P(int n, int a, int b, const DataType& x, DataType* out)
//    {
//        // From spectral methods page 351.
//        out[0] = DataType(1.0);
//
//        if( n >= 1 )
//        {
//            out[1] = .5*((ElVisFloat)a-(ElVisFloat)b+((ElVisFloat)a+(ElVisFloat)b+MAKE_FLOAT(2.0))*x);
//        }
//        
//        ElVisFloat apb = a+b;
//        ElVisFloat alpha = a;
//        ElVisFloat beta = b;
//        for(int k = 2; k <= n; ++k)
//        {
//            ElVisFloat a1 = MAKE_FLOAT(2.0)*k*(k + apb)*(MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(2.0));
//            ElVisFloat a2 = (MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(1.0))*(alpha*alpha - beta*beta);
//            ElVisFloat a3 = (MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(2.0))*(MAKE_FLOAT(2.0)*k + apb - MAKE_FLOAT(1.0))*(MAKE_FLOAT(2.0)*k + apb);
//            ElVisFloat a4 = MAKE_FLOAT(2.0)*(k + alpha - MAKE_FLOAT(1.0))*(k + beta - MAKE_FLOAT(1.0))*(MAKE_FLOAT(2.0)*k + apb);
//
//            a2 /= a1;
//            a3 /= a1;
//            a4 /= a1;
//
//            out[k] = (a2 + a3*x)*out[k-1] - a4*out[k-2];
//        }
//    }
//
//    template<typename DataType>
//    __device__ __forceinline__ DataType dP(int n, int a, int b, const DataType& x)
//    {
//        if( n != 0 )
//        {
//            return MAKE_FLOAT(.5)*(n+a+b+1)*P(n-1, a+1, b+1, x);
//        }
//        return 0;
//    }
//
//    template<typename DataType>
//    __device__ __forceinline__ DataType ddP(int n, int a, int b, const DataType& x)
//    {
//        if( n >= 2 )
//        {
//            return MAKE_FLOAT(.25)*(n+a+b+1)*(n+a+b+2)*P(n-2, a+2, b+2, x);
//        }
//        return DataType(0);
//    }
//}

#endif //ELVIS_JACOBI_CU
