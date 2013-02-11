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

#ifndef ELVIS_MATH_POLYNOMIAL_PROJECTIONS_HPP
#define ELVIS_MATH_POLYNOMIAL_PROJECTIONS_HPP

#include <ElVis/Math/Polynomial.hpp>
#include <ElVis/Math/GaussLegendreQuadrature.hpp>

namespace ElVis
{
    namespace Math
    {
        template<typename Type>
        Polynomial<Type, OrthogonalLegendreBasis<Type> >
        GenerateLeastSquaresPolynomialProjection(unsigned int projectionOrder,
                                                 unsigned int integrationOrder,
                                                 const boost::function1<Type, Type>& f)
        {
            std::vector<Type> coeffs(projectionOrder+1, 0);

            const std::vector<Type>* nodes;
            const std::vector<Type>* weights;
            const unsigned int n = integrationOrder;

            GaussLegendreNodesAndWeights<double>::GenerateGaussLegendreNodesAndWeights(
                nodes, weights, n+1);

            std::vector<Type> vals(n);
            for(unsigned int j = 0; j < n; ++j)
            {
                vals[j] = f((*nodes)[j]);
            }

            for(unsigned int c_index = 0; c_index <= projectionOrder; ++c_index)
            {
                coeffs[c_index] = 0.0;
                for(unsigned int k = 0; k < n; ++k)
                {
                    coeffs[c_index] += vals[k] * OrthogonalLegendreBasis<Type>::eval(c_index, (*nodes)[k]) *
                        (*weights)[k];
                }

            }

            return Polynomial<Type, OrthogonalLegendreBasis<Type> >(coeffs);
        }
    }
}

#endif
