////////////////////////////////////////////////////////////////////////////////
//
//  File: hoJacobi.h
//
//
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

#ifndef ELVIS_JACOBI_EXTENSION___JACOBI_HPP__
#define ELVIS_JACOBI_EXTENSION___JACOBI_HPP__

//#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>

//#include <iostream>
//#include <math.h>

// Jacobi polynomials and derivatives.
// Reference the book.
// Note that this file contains mostly generated code and
// therefore care should be taken if modifications are needed.
namespace Jacobi
{
    // Evaluates the nth Jacobi polynomial at point x.
    template<typename DataType>
        DataType Legendre(int n, const DataType& x);

    template<typename DataType>
        DataType P(int n, int a, int b, const DataType& x);

    template<typename DataType>
        DataType dP(int n, int a, int b, const DataType& x);

    template<typename DataType>
        DataType ddP(int n, int a, int b, const DataType& x);
}

namespace Jacobi
{
    template<typename DataType>
        DataType Legendre(int n, const DataType& x)
    {
        return P(n, 0, 0, x);
    }

    // Evaluates the nth Jacobi polynomial at point x.
    template<typename DataType>
    DataType P(int n, int a, int b, const DataType& x)
    {
        // From spectral methods page 351.
        if( n == 0 )
        {
            return DataType(1.0);
        }

        if( n == 1 )
        {
            return .5*((double)a-(double)b+((double)a+(double)b+2.0)*x);
        }

        DataType result(0.0);
        double apb = a + b;

        DataType polyn2(1.0);
        DataType polyn1 = 0.5*( (double)a - (double)b + ((double)a + (double)b + 2.0)*x);
        double alpha = a;
        double beta = b;
        for(int k = 2; k <= n; ++k)
        {
            double a1 = 2.0*k*(k + apb)*(2.0*k + apb - 2.0);
            double a2 = (2.0*k + apb - 1.0)*(alpha*alpha - beta*beta);
            double a3 = (2.0*k + apb - 2.0)*(2.0*k + apb - 1.0)*(2.0*k + apb);
            double a4 = 2.0*(k + alpha - 1.0)*(k + beta - 1.0)*(2.0*k + apb);

            a2 /= a1;
            a3 /= a1;
            a4 /= a1;

            //DataType t1 = a4*polyn2;
            //DataType t2 = (a2+a3*x);
            //DataType t3 = t2*polyn1;
            result = (a2 + a3*x)*polyn1 - a4*polyn2;

            polyn2 = polyn1;
            polyn1 = result;
        }


        return result;
    }

    template<typename DataType>
        DataType dP(int n, int a, int b, const DataType& x)
    {
        if( n != 0 )
        {
            return .5*(n+a+b+1)*P(n-1, a+1, b+1, x);
        }
        return 0;
    }

    template<typename DataType>
        DataType ddP(int n, int a, int b, const DataType& x)
    {
        if( n >= 2 )
        {
            return .25*(n+a+b+1)*(n+a+b+2)*P(n-2, a+2, b+2, x);
        }
        return DataType(0);
    }
}

#endif
