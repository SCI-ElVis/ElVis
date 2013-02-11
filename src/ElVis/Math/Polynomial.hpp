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
 

#ifndef ELVIS_MATH_POLYNOMIAL_HPP
#define ELVIS_MATH_POLYNOMIAL_HPP

#include <vector>
#include <ElVis/Math/Jacobi.hpp>
#include <string>
#include <math.h>

namespace ElVis
{
    namespace Math
    {
        template<typename Type, typename BasisType>
        class Polynomial
        {
            public:
                Polynomial() :
                    m_coefficients()
                {
                    m_coefficients.push_back(0);
                }

                Polynomial(const std::vector<Type>& coeffs) :
                    m_coefficients(coeffs)
                {
                    if( m_coefficients.size() == 0 )
                    {
                        throw std::string("Error: Empty polynomial.");
                    }
                }

                Polynomial(const Polynomial& rhs) :
                    m_coefficients(rhs.m_coefficients)
                {
                }

                Polynomial& operator=(const Polynomial& rhs)
                {
                    m_coefficients = rhs.m_coefficients;
                    return *this;
                }

                virtual ~Polynomial() {}

                template<typename T>
                T operator()(const T& x) const
                {
                    T result(0.0);
                    for(unsigned int i = 0; i <= degree(); ++i)
                    {
                        result += BasisType::eval(i, x) * coeff(i);
                    }
                    return result;
                }

                unsigned int degree() const
                {
                    return static_cast<unsigned int>(m_coefficients.size()-1);
                }

                Type coeff(unsigned int index) const
                {
                    return m_coefficients[index];
                }

                Type& coeff(unsigned int index)
                {
                    return m_coefficients[index];
                }

                const std::vector<Type> coeffs() const
                {
                    return m_coefficients;
                }

            private:
                std::vector<Type> m_coefficients;
        };

        template<typename Type>
        class MonomialBasis
        {
            public:
                static Type eval(unsigned int i, const Type& x)
                {
                    Type result = 1.0;
                    for(unsigned int index = 0; index < i; ++index)
                    {
                        result *= x;
                    }

                    return result;
                }
                //static void write(std::ostream& os, unsigned int i);
        };

        template<typename Type>
        class LegendreBasis
        {
            public:
                template<typename T>
                static T eval(unsigned int i, const T& x)
                {
                    return OrthoPoly::P(i, 0, 0, x);
                }
                //static void write(std::ostream& os, unsigned int i);
        };

        template<typename Type>
        class OrthogonalLegendreBasis
        {
            public:
                static Type eval(unsigned int i, const Type& x)
                {
                    return sqrt((2.0*i+1.0)/2.0) * OrthoPoly::P(i, 0, 0, x);
                }
                //static void write(std::ostream& os, unsigned int i);
        };

        typedef Polynomial<double, MonomialBasis<double> > Monomial;
        typedef Polynomial<double, LegendreBasis<double> > LegendrePolynomial;
        typedef Polynomial<double, OrthogonalLegendreBasis<double> > OrthogonalLegendrePolynomial;
    }
}

#endif
