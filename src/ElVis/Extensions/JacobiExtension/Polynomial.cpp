////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolynomial.cpp
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
////////////////////////////////////////////////////////////////////////////////
#include <ElVis/Extensions/JacobiExtension/Isosurface.h>
#include <ElVis/Extensions/JacobiExtension/Polynomial.hpp>
#include <ElVis/Extensions/JacobiExtension/FiniteElementMath.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::cout;

namespace Polynomial
{
    std::vector<double> findRealRoots(const Monomial& poly)
    {
        #ifdef LAPACK_ENABLED
            return findRealRootsWithCLapack(poly);
        #else
            /// \todo Make this generic.
            std::vector<double> result;
            ElVis::JacobiExtension::findRealRoots(poly.coeffs(),
                poly.degree(), result);
            return result;
        #endif
    }

    #ifdef LAPACK_ENABLED
        std::vector<double> findRealRootsWithCLapack(const Monomial& poly)
        {
            // \todo - Hard coded 1st and second order.
            Monomial reducedPoly = poly;
            reducedPoly.reducePolynomialByRemovingSmallLeadingCoefficients(
                std::numeric_limits<double>::epsilon());
            cout << "Original Polynomial: " << poly << endl;
            cout << "New polynomial: " << reducedPoly << endl;
            // Construct the hessenberg matrix which will be used
            // for root finding.
            unsigned int matrix_size = reducedPoly.degree()*reducedPoly.degree();
            double* hess_matrix = new double[matrix_size];
            memset(hess_matrix, 0, sizeof(double)*matrix_size);

            // Set the ones.
            for(unsigned int i = 0; i < reducedPoly.degree()-1; ++i)
            {
                hess_matrix[i*reducedPoly.degree() + i + 1] = 1.0;
            }

            // Now set the coefficient values.
            for(unsigned int i = matrix_size-reducedPoly.degree(); i < matrix_size; ++i)
            {
                unsigned int current_coeff = i - (matrix_size-reducedPoly.degree());
                hess_matrix[i] = -reducedPoly.coeff(current_coeff)/reducedPoly.coeff(reducedPoly.degree());
            }

//          cout << "original matrix." << endl;
//          for(unsigned int i = 0; i < matrix_size; ++i)
//          {
//              cout << hess_matrix[i] << endl;
//          }
            // Now balance the matrix.
            // Scale and permute the matrix.
            char balanceJob = 'B';
            integer n = reducedPoly.degree();
            integer ilo = 1;
            integer ihi = n;
            integer info = 0;
            double* scale = new double[matrix_size];

            dgebal_(&balanceJob, &n, hess_matrix,
                &n, &ilo, &ihi, scale, &info);
            if( info != 0 )
            {
                cerr << "Error computing balanced matrix." << endl;
            }
//          cout << "Balanced matrix." << endl;
//          for(unsigned int i = 0; i < matrix_size; ++i)
//          {
//              cout << hess_matrix[i] << endl;
//          }
            delete [] scale;

            // Now that the matrix is balanced, find the roots.
            char rootJob = 'E';
            char compz = 'N';
            double* wr = new double[reducedPoly.degree()];
            double* wi = new double[reducedPoly.degree()];
            integer work_size = matrix_size*10;
            double* work = new double[work_size];

            dhseqr_(&rootJob, &compz, &n, &ilo, &ihi, hess_matrix, &n,
                wr, wi, NULL, &n, work, &work_size, &info);

            cout << "Root finding matrix." << endl;
//          for(unsigned int i = 0; i < matrix_size; ++i)
//          {
//              cout << hess_matrix[i] << endl;
//          }
            if( info < 0 )
            {
                cerr << "Error computing the " << -info << " argument in dhseqr" << endl;
            }
            else if( info > 0 )
            {
                cerr << "Not all eigenvalues were computed. i = " << info << endl;
            }

            std::vector<double> result;
            for(unsigned int i = 0; i < reducedPoly.degree(); ++i )
            {
//              cout << "wr[" << i << "] = " << wr[i] << endl;
//              cout << "wi[" << i << "] = " << wi[i] << endl;
                if( wi[i] == 0.0 )
                {
                    result.push_back(wr[i]);
                }
            }

            delete [] wr;
            delete [] wi;
            delete [] work;
            delete [] hess_matrix;
            return result;
        }
    #endif
};

