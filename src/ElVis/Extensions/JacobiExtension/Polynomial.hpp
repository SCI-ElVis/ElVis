////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolynomial.hpp
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

#ifndef ELVIS_JACOBI_EXTENSION___POLYNOMIAL_H__
#define ELVIS_JACOBI_EXTENSION___POLYNOMIAL_H__

#include <boost/function.hpp>
#include <vector>
#include <math.h>
#include <ElVis/Extensions/JacobiExtension/Jacobi.hpp>
#include <iostream>

#include <ElVis/Extensions/JacobiExtension/Declspec.h>

#ifdef USE_CLAPACK
#ifdef __MINGW32_VERSION
#ifdef small
#undef small
#endif
#endif
#include <clapack.h>
#define LAPACK_ENABLED
#endif

#ifdef USE_LAPACK
#include <lapack.h>
#define LAPACK_ENABLED
#endif

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace Polynomial
{
    template<typename Type, typename BasisType>
    class Polynomial
    {
    public:
        Polynomial();
        Polynomial(const std::vector<Type>& coeffs);
        Polynomial(const Polynomial& rhs);
        Polynomial& operator=(const Polynomial& rhs);
        virtual ~Polynomial();
        virtual Type operator()(const Type& x) const;

        unsigned int degree() const;
        Type coeff(unsigned int index) const;
        Type& coeff(unsigned int index);

        const std::vector<Type> coeffs() const;
        unsigned int reducePolynomialByRemovingSmallLeadingCoefficients(double epsilon);
    private:
        std::vector<Type> m_coefficients;
    };

    template<typename Type, typename BasisType>
    std::ostream& operator<<(std::ostream& os, const Polynomial<Type, BasisType>& poly);

    template<typename Type>
    class MonomialBasis
    {
    public:
        static Type eval(unsigned int i, const Type& x);
        static void write(std::ostream& os, unsigned int i);
    };

    template<typename Type>
    class LegendreBasis
    {
    public:
        static Type eval(unsigned int i, const Type& x);
        static void write(std::ostream& os, unsigned int i);
    };

    template<typename Type>
    class OrthogonalLegendreBasis
    {
    public:
        static Type eval(unsigned int i, const Type& x);
        static void write(std::ostream& os, unsigned int i);
    };

    /// \todo We may want a monomial type which uses Horner's algorithm.
    typedef Polynomial<double, MonomialBasis<double> > Monomial;
    typedef Polynomial<double, LegendreBasis<double> > LegendrePolynomial;
    typedef Polynomial<double, OrthogonalLegendreBasis<double> > OrthogonalLegendrePolynomial;

    template<typename Type>
    Polynomial<Type, MonomialBasis<Type> > convertToMonomial(
        const Polynomial<Type, OrthogonalLegendreBasis<Type> >& poly);

    /// \todo Template this.
    JACOBI_EXTENSION_EXPORT std::vector<double> findRealRoots(const Monomial& poly);
#ifdef LAPACK_ENABLED
    std::vector<double> findRealRootsWithCLapack(const Monomial& poly);
#endif
}

namespace Polynomial
{
    template<typename Type, typename BasisType>
    Polynomial<Type, BasisType>::Polynomial() :
    m_coefficients()
    {
        m_coefficients.push_back(0);
    }

    template<typename Type, typename BasisType>
    Polynomial<Type, BasisType>::Polynomial(const std::vector<Type>& coeffs) :
    m_coefficients(coeffs)
    {
        if( m_coefficients.size() == 0 )
        {
            throw std::string("Error: Empty polynomial.");
        }
    }

    template<typename Type, typename BasisType>
    Polynomial<Type, BasisType>::Polynomial(const Polynomial<Type,BasisType>& rhs) :
    m_coefficients(rhs.m_coefficients)
    {
    }

    template<typename Type, typename BasisType>
    Polynomial<Type, BasisType>& Polynomial<Type, BasisType>::operator=(const Polynomial<Type, BasisType>& rhs)
    {
        m_coefficients = rhs.m_coefficients;
        return *this;
    }

    template<typename Type, typename BasisType>
    Polynomial<Type, BasisType>::~Polynomial()
    {
    }

    template<typename Type, typename BasisType>
    unsigned int Polynomial<Type, BasisType>::degree() const
    {
        return static_cast<unsigned int>(m_coefficients.size()-1);
    }

    template<typename Type, typename BasisType>
    Type Polynomial<Type, BasisType>::coeff(unsigned int index) const
    {
        return m_coefficients[index];
    }

    template<typename Type, typename BasisType>
    Type& Polynomial<Type, BasisType>::coeff(unsigned int index)
    {
        return m_coefficients[index];
    }

    template<typename Type, typename BasisType>
    const std::vector<Type> Polynomial<Type, BasisType>::coeffs() const
    {
        return m_coefficients;
    }

    template<typename Type, typename BasisType>
    Type Polynomial<Type, BasisType>::operator()(const Type& x) const
    {
        Type result = 0.0;
        for(unsigned int i = 0; i <= degree(); ++i)
        {
            result += BasisType::eval(i, x) * coeff(i);
        }
        return result;
    }

    template<typename Type, typename BasisType>
    unsigned int Polynomial<Type, BasisType>::reducePolynomialByRemovingSmallLeadingCoefficients(double epsilon)
    {
        for(unsigned int i = degree(); i >= 1; --i)
        {
            if( fabs(coeff(i)) > epsilon )
            {
                m_coefficients.resize(i+1);
                return i;
            }
        }

        // Kind of a bizarre case.  What should we do here?
        return degree();
    }

    template<typename Type, typename BasisType>
    std::ostream& operator<<(std::ostream& os, const Polynomial<Type, BasisType>& poly)
    {
        int prevPrecision = os.precision();
        os.precision(20);
        //fmtflags prevFlags = os.setf(std::ios::showpos);

        for(unsigned int i = poly.degree(); i >= 1; --i)
        {
            os << poly.coeff(i) << "*";
            BasisType::write(os, i);
            os << " ";
        }
        os << poly.coeff(0);

        os.precision(prevPrecision);
        //os.setf(prevFlags);
        return os;
    }

    template<typename Type>
    Type MonomialBasis<Type>::eval(unsigned int i, const Type& x)
    {
        Type result = 1.0;
        for(unsigned int index = 0; index < i; ++index)
        {
            result *= x;
        }

        return result;
    }

    template<typename Type>
    void MonomialBasis<Type>::write(std::ostream& os, unsigned int i)
    {
        os << "t^" << i;
    }

    template<typename Type>
    Type LegendreBasis<Type>::eval(unsigned int i, const Type& x)
    {
        return Jacobi::P(i, 0, 0, x);
    }

    template<typename Type>
    void LegendreBasis<Type>::write(std::ostream& os, unsigned int i)
    {
        os << "P(" << i << ", 0, 0, \'t\')";
    }

    template<typename Type>
    Type OrthogonalLegendreBasis<Type>::eval(unsigned int i, const Type& x)
    {
        return sqrt((2.0*i+1.0)/2.0) * Jacobi::P(i, 0, 0, x);
    }

    template<typename Type>
    void OrthogonalLegendreBasis<Type>::write(std::ostream& os, unsigned int i)
    {
        os << "OrthoP(" << i << ", 0, 0, \'t\')";
    }

    template<typename Type>
    std::vector<double> generateMonomialCoefficients(unsigned int order,
        const Polynomial<Type, OrthogonalLegendreBasis<Type> >& poly)
    {
        std::vector<double> coeffs(order+1, 0.0);

        if( order == 0 )
        {
            coeffs[0] = .70710678118654752440 * poly.coeff(0);
            return coeffs;
        }
        else if( order == 1 )
        {
            coeffs[0] = .70710678118654752440*poly.coeff(0);
            coeffs[1] = 1.2247448713915890491*poly.coeff(1);
            return coeffs;
        }

        /// \todo I could build a static table upt to a certain
        /// order if this proves to be a bottleneck.
        std::vector<std::vector<double> > coeffTable;
        coeffTable.resize(order+1);
        for(unsigned int i = 0; i < order+1; ++i)
        {
            coeffTable[i].assign(order+1, 0.0);
        }

        coeffTable[0][0] = 1.0;
        coeffTable[1][0] = 0.0;
        coeffTable[1][1] = 1.0;
        for(unsigned int i = 2; i < order+1; ++i)
        {
            double n = i-1;
            coeffTable[i][0] = -n/(n+1.0) * coeffTable[i-2][0];
            for(unsigned int j = 1; j < order+1; ++j)
            {
                coeffTable[i][j] = (2.0*n+1.0)/(n+1.0)*coeffTable[i-1][j-1] -
                    n/(n+1.0)*coeffTable[i-2][j];
            }
        }

        // Now that the table has been generated, apply the factor which
        // makes it orthogonal.
        for(unsigned int i = 0; i <= order; ++i)
        {
            double factor = sqrt((2.0*static_cast<double>(i)+1.0)/2.0);
            for(unsigned int j = 0; j <= order; ++j)
            {
                coeffTable[i][j] *= factor;
            }
        }

        // Now that we have the coefficient table we can convert.
        for(unsigned int coeffIndex = 0; coeffIndex <= order; ++coeffIndex)
        {
            for(unsigned int legCoeffIndex = 0; legCoeffIndex <= order; ++legCoeffIndex)
            {
                coeffs[coeffIndex] += poly.coeff(legCoeffIndex)*
                    coeffTable[legCoeffIndex][coeffIndex];
            }
        }

        return coeffs;
    }

    template<typename Type>
    Polynomial<Type, MonomialBasis<Type> > convertToMonomial(
        const Polynomial<Type, OrthogonalLegendreBasis<Type> >& poly)
    {
        std::vector<double> coeffs = generateMonomialCoefficients(poly.degree(), poly);

        return Polynomial<Type, MonomialBasis<Type> >(coeffs);
    }
}



#endif

