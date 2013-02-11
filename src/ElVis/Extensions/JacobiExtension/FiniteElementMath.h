////////////////////////////////////////////////////////////////////////////////
//
//  File: hoFiniteElementMath.h
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
//  This file contains many of the linear algebra and assorted other
//  math routines necessary for ray tracing finite element volumes.
//
////////////////////////////////////////////////////////////////////////////////


#ifndef ELVIS_JACOBI_EXTENSION__FINITE_ELEMENT_MATH_H_
#define ELVIS_JACOBI_EXTENSION__FINITE_ELEMENT_MATH_H_

#include <vector>
#include <set>


#include <ElVis/Core/Vector.hpp>
#include <ElVis/Core/Point.hpp>
#include <vector>

#include <ElVis/Extensions/JacobiExtension/Declspec.h>

namespace ElVis
{
    namespace JacobiExtension
    {
        // Calculates the chebyshev points in the interval between a and b.
        // n represents the degree of the polynomial we are trying to interpolate,
        // so this routine will obtain n+1 sample points.
        //
        // Returns true if the points were calculated, false otherwise.
        // The points are returned in the result array, which must have a size
        // of n+1.
        JACOBI_EXTENSION_EXPORT bool getChebyshevPoints(double a, double b, int n, double* result);
        JACOBI_EXTENSION_EXPORT bool getChebyshevPoints(double a, double b, int n, std::set<double>& result);

        // Given the sample points in x and the scalar values in y,
        // find the coefficients of the interpolating polynomial of degree
        // n.  The coefficients are returned in a.
        // x, y, and a must have n+1 entries.
        JACOBI_EXTENSION_EXPORT bool generateInterpolatingPolynomial(const double* x,
            const double* y,
            int n,
            double* a);


        // Finds the roots of the given polynomial.
        // n is the order, so coefficients and roots must have
        // n+1 entries.
        // The return value is the number of real roots found.
        // Finds the roots of the polynomial with the given coefficients.
        //
        // So for polynomial a_0 + a_1 * x + a_2 * x^2 + ... a_n * x^n
        //
        // coefficients[0] = a_0
        // coefficients[1] = a_1
        // coefficients[n] = a_n
        //
        // The order parameters tells us the order of the polynomial.
        // The size of the coefficients and roots arrays must be
        // order+1.
        //
        // Returns the number of real roots found.  Imaginary roots
        // are ignored.
        //
        JACOBI_EXTENSION_EXPORT int findRealRoots(const double* coefficients,
            int n, double* roots);
        JACOBI_EXTENSION_EXPORT int findRealRoots(const std::vector<double>& coefficients,
            int n, std::vector<double>& roots);

        //Finds all eigenvalues of an upper Hessenberg matrix a[1..n][1..n]. On input a can be
        //exactly as output from elmhes �1.5; on output it is destroyed. The real and imaginary parts
        //of the eigenvalues are returned in wr[1..n] and wi[1..n], respectively.
        // From Numerical Recipes.
        JACOBI_EXTENSION_EXPORT void hqr(double **a, int n, double* wr, double* wi);

        // Balances the matrix.  This is from NR and is a full balancer, I should be able
        // to derive a faster balancer based on the hessenberg matrix structure.
        JACOBI_EXTENSION_EXPORT void balanc(double** a, int n);

        JACOBI_EXTENSION_EXPORT void myBalance(double** a, int n);

        // Floating point values are divided into signs, mantissas, and exponents.
        // This function returns the value of the exponent.
        JACOBI_EXTENSION_EXPORT int getExponent(double val);
        JACOBI_EXTENSION_EXPORT double setExponent(double val, int exponent);
        JACOBI_EXTENSION_EXPORT double shiftExponent(double val, int shift);


        // Creates a hessenberg matrix with the given coefficients and stores the
        // resulting matrix in result.  This matrix is balanced.
        // The matrix is suitable for root finding in hqr.  This means that the matrix
        // starts on element 1 in both rows and columns.  So all entries result[0][y] and
        // result[x][0] are necessary but unused.
        JACOBI_EXTENSION_EXPORT void generateRowMajorHessenbergMatrix(const double* coefficients, int n, double** result);

        JACOBI_EXTENSION_EXPORT void generateCoordinateSystemFromVector(const ElVis::WorldVector& basis, ElVis::WorldVector& u,
            ElVis::WorldVector& v, ElVis::WorldVector& w);

        // These are temporary functions while I explore using LAPACK again.
        // Once we decide on the eigensolver I'll make the appropriate changes directly in the
        // hessenberg matrix generator.
        JACOBI_EXTENSION_EXPORT void swapMatrixOrder(double** h, int n);

        // The numerical recipes code is 1 indexed, LAPACK is 0 indexed.
        //
        JACOBI_EXTENSION_EXPORT void makeMatrix0Indexed(double** h, int n);

        JACOBI_EXTENSION_EXPORT void collapseMatrix(double* h, double** result, int n);

        //JACOBI_EXTENSION_EXPORT bool planeIntersection(const rt::Ray& theRay, const ElVis::WorldVector& normal, double D, double &t);
        JACOBI_EXTENSION_EXPORT int numberOfTimesEdgeCrossesPositiveXAxis(const ElVis::WorldPoint& g1, const ElVis::WorldPoint& g2);

        JACOBI_EXTENSION_EXPORT double evalPolynomial(double* coeffs, double x, int n);
        JACOBI_EXTENSION_EXPORT double evalPolynomialDerivative(double* coeffs, double x, int n);

        JACOBI_EXTENSION_EXPORT double factorial(unsigned int n);
        JACOBI_EXTENSION_EXPORT double binomial(unsigned int n, unsigned int k);
    }
}

#endif
