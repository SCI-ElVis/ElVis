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

#ifndef ELVIS_MATH_JACOBI_HPP
#define ELVIS_MATH_JACOBI_HPP

#include <ElVis/Core/Float.h>
#include <ElVis/Core/Cuda.h>

namespace ElVis
{
  namespace OrthoPoly
  {
    // Evaluates the nth order Legendre polynomial at point x.
    template <typename DataType>
    DataType Legendre(int n, const DataType& x);

    template <typename DataType>
    ELVIS_DEVICE DataType P(int n, int a, int b, const DataType& x);

    template <typename DataType>
    ELVIS_DEVICE DataType dP(int n, int a, int b, const DataType& x);

    template <typename DataType>
    ELVIS_DEVICE DataType ddP(int n, int a, int b, const DataType& x);

    template <typename DataType>
    DataType Legendre(int n, const DataType& x)
    {
      return P(n, 0, 0, x);
    }

    template <int n>
    struct EvalJacobiPolynomial;

    template <>
    struct EvalJacobiPolynomial<0>
    {
      template <typename DataType>
      static ELVIS_DEVICE DataType Evaluate(int a, int b, const DataType& x)
      {
        return DataType(MAKE_FLOAT(1.0));
      }
    };

    template <>
    struct EvalJacobiPolynomial<1>
    {
      template <typename DataType>
      static ELVIS_DEVICE DataType Evaluate(int a, int b, const DataType& x)
      {
        return MAKE_FLOAT(.5) *
               ((ElVisFloat)a - (ElVisFloat)b +
                ((ElVisFloat)a + (ElVisFloat)b + MAKE_FLOAT(2.0)) * x);
      }
    };

    template <int n>
    struct EvalJacobiPolynomial
    {
      template <typename DataType>
      static ELVIS_DEVICE DataType Evaluate(int a, int b, const DataType& x)
      {
        ElVisFloat denom = 2 * n * (n + a + b) * (2 * n + a + b - 2);
        DataType c0 =
          (2 * n + a + b - 1) *
          ((2 * n + a + b) * (2 * n + a + b - 2) * x + a * a + b * b);
        ElVisFloat c1 = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b);
        DataType numerator =
          c0 * EvalJacobiPolynomial<n - 1>::Evaluate(a, b, x) -
          c1 * EvalJacobiPolynomial<n - 2>::Evaluate(a, b, x);
        return numerator / denom;
      }
    };

    // Evaluates the nth Jacobi polynomial at point x.
    ELVIS_DEVICE ElVisFloat P(int n, int a, int b, const ElVisFloat& x)
    {
      // From spectral methods page 351.
      switch (n)
      {
        case 0:
          return EvalJacobiPolynomial<0>::Evaluate(a, b, x);
        case 1:
          return EvalJacobiPolynomial<1>::Evaluate(a, b, x);
        case 2:
          return EvalJacobiPolynomial<2>::Evaluate(a, b, x);
        case 3:
          return EvalJacobiPolynomial<3>::Evaluate(a, b, x);
        case 4:
          return EvalJacobiPolynomial<4>::Evaluate(a, b, x);
        case 5:
          return EvalJacobiPolynomial<5>::Evaluate(a, b, x);
        case 6:
          return EvalJacobiPolynomial<6>::Evaluate(a, b, x);
        case 7:
          return EvalJacobiPolynomial<7>::Evaluate(a, b, x);
        case 8:
          return EvalJacobiPolynomial<8>::Evaluate(a, b, x);
        case 9:
          return EvalJacobiPolynomial<9>::Evaluate(a, b, x);
        case 10:
          return EvalJacobiPolynomial<10>::Evaluate(a, b, x);
        case 11:
          return EvalJacobiPolynomial<11>::Evaluate(a, b, x);
        default:
          break;
      }

      // If we haven't specialized, then default to the original code.
      ElVisFloat result(MAKE_FLOAT(0.0));
      ElVisFloat apb = a + b;

      ElVisFloat polyn2(MAKE_FLOAT(1.0));
      ElVisFloat polyn1 =
        MAKE_FLOAT(0.5) *
        ((ElVisFloat)a - (ElVisFloat)b +
         ((ElVisFloat)a + (ElVisFloat)b + MAKE_FLOAT(2.0)) * x);
      ElVisFloat alpha = a;
      ElVisFloat beta = b;
      for (int k = 2; k <= n; ++k)
      {
        ElVisFloat a1 = MAKE_FLOAT(2.0) * k * (k + apb) *
                        (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(2.0));
        ElVisFloat a2 = (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(1.0)) *
                        (alpha * alpha - beta * beta);
        ElVisFloat a3 = (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(2.0)) *
                        (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(1.0)) *
                        (MAKE_FLOAT(2.0) * k + apb);
        ElVisFloat a4 = MAKE_FLOAT(2.0) * (k + alpha - MAKE_FLOAT(1.0)) *
                        (k + beta - MAKE_FLOAT(1.0)) *
                        (MAKE_FLOAT(2.0) * k + apb);

        a2 /= a1;
        a3 /= a1;
        a4 /= a1;

        result = (a2 + a3 * x) * polyn1 - a4 * polyn2;

        polyn2 = polyn1;
        polyn1 = result;
      }

      return result;
    }

    template <typename DataType>
    ELVIS_DEVICE void P(int n, int a, int b, const DataType& x, DataType* out)
    {
      out[0] = DataType(MAKE_FLOAT(1.0));

      if (n >= 1)
      {
        out[1] = MAKE_FLOAT(.5) *
                 ((ElVisFloat)a - (ElVisFloat)b +
                  ((ElVisFloat)a + (ElVisFloat)b + MAKE_FLOAT(2.0)) * x);
      }

      ElVisFloat apb = a + b;
      ElVisFloat alpha = a;
      ElVisFloat beta = b;
      for (int k = 2; k <= n; ++k)
      {
        ElVisFloat a1 = MAKE_FLOAT(2.0) * k * (k + apb) *
                        (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(2.0));
        ElVisFloat a2 = (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(1.0)) *
                        (alpha * alpha - beta * beta);
        ElVisFloat a3 = (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(2.0)) *
                        (MAKE_FLOAT(2.0) * k + apb - MAKE_FLOAT(1.0)) *
                        (MAKE_FLOAT(2.0) * k + apb);
        ElVisFloat a4 = MAKE_FLOAT(2.0) * (k + alpha - MAKE_FLOAT(1.0)) *
                        (k + beta - MAKE_FLOAT(1.0)) *
                        (MAKE_FLOAT(2.0) * k + apb);

        a2 /= a1;
        a3 /= a1;
        a4 /= a1;

        out[k] = (a2 + a3 * x) * out[k - 1] - a4 * out[k - 2];
      }
    }

    template <typename DataType>
    ELVIS_DEVICE DataType dP(int n, int a, int b, const DataType& x)
    {
      if (n != 0)
      {
        return .5 * (n + a + b + 1) * P(n - 1, a + 1, b + 1, x);
      }
      return 0;
    }

    template <typename DataType>
    ELVIS_DEVICE void dP(int n, int a, int b, const DataType& x, DataType* out)
    {
      for (int i = 0; i <= n; ++i)
      {
        out[i] = dP(i, a, b, x);
      }
    }

    template <typename DataType>
    ELVIS_DEVICE DataType ddP(int n, int a, int b, const DataType& x)
    {
      if (n >= 2)
      {
        return .25 * (n + a + b + 1) * (n + a + b + 2) *
               P(n - 2, a + 2, b + 2, x);
      }
      return DataType(0);
    }
  }
}

#endif
