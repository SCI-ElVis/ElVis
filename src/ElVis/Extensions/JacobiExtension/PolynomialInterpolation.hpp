////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolynomialInterpolation.h
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

#ifndef ELVIS_JACOBI_EXTENSION___POLYNOMIAL_INTERPOLATION_H__
#define ELVIS_JACOBI_EXTENSION___POLYNOMIAL_INTERPOLATION_H__

#include <ElVis/Extensions/JacobiExtension/Polynomial.hpp>
#include <ElVis/Extensions/JacobiExtension/NumericalIntegration.hpp>
#include <boost/bind.hpp>

namespace PolynomialInterpolation
{
    /// Generates a least squares polynomial projects for f.
    /// \param order The order of the projected polynomial.
    /// \param f The function to be fit.
    template<typename Type>
    Polynomial::Polynomial<Type, Polynomial::OrthogonalLegendreBasis<Type> >
    generateLeastSquaresPolynomialProjection(unsigned int order,
        const boost::function1<Type, Type>& f);
};

namespace PolynomialInterpolation
{
    template<typename Type>
    class MultiplyFunctions
    {
        public:
            MultiplyFunctions(const boost::function1<Type, Type>& lhs,
                const boost::function1<Type, Type>& rhs) :
                    m_lhs(lhs),
                    m_rhs(rhs)
                    {
                    }

            Type operator()(const Type& x)
            {
                return m_lhs(x)*m_rhs(x);
            }

        private:
            boost::function1<Type, Type> m_lhs;
            boost::function1<Type, Type> m_rhs;
    };

    template<typename Type>
    Polynomial::Polynomial<Type, Polynomial::OrthogonalLegendreBasis<Type> >
    generateLeastSquaresPolynomialProjection(unsigned int order,
        const boost::function1<Type, Type>& f)
    {
        std::vector<Type> coeffs(order+1, 0);

        const std::vector<Type>* nodes;
        const std::vector<Type>* weights;
        const unsigned int n = 2*order+6;

        NumericalIntegration::GaussLegendreNodesAndWeights<double>::GenerateGaussLegendreNodesAndWeights(
            nodes, weights, n);

        std::vector<Type> vals(n);
        for(unsigned int j = 0; j < n; ++j)
        {
            vals[j] = f((*nodes)[j]);
        }

        for(unsigned int c_index = 0; c_index <= order; ++c_index)
        {
            coeffs[c_index] = 0.0;
            for(unsigned int k = 0; k < n; ++k)
            {
                coeffs[c_index] += vals[k] * Polynomial::OrthogonalLegendreBasis<Type>::eval(c_index, (*nodes)[k]) *
                    (*weights)[k];
            }

            /*
            MultiplyFunctions<double> integral(f,
                boost::bind(Polynomial::OrthogonalLegendreBasis<double>::eval, c_index, _1));
            boost::function1<double, double> integral_func = integral;
            coeffs[c_index] = NumericalIntegration::Trapezoidal(integral_func,
                -1.0, 1.0, 1000000);
            */
//          MultiplyFunctions<double> integral(f,
//              boost::bind(Polynomial::OrthogonalLegendreBasis<double>::eval, j, _1));
//          boost::function1<double, double> integral_func = integral;
//          coeffs[j] = NumericalIntegration::GaussLegendreQuadrature(integral_func,
//              5*order+6, -1.0, 1.0);
        }

        return Polynomial::Polynomial<Type, Polynomial::OrthogonalLegendreBasis<Type> >(coeffs);
    }
}


#endif
