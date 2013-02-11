////////////////////////////////////////////////////////////////////////////////
//
//  File: hoPolylib.cpp
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
/**************************************************************************/
//                                                                        //
//   Author:    S.Sherwin                                                 //
//   Design:    S.Sherwin                                                 //
//   Date  :    12/4/96                                                   //
//                                                                        //
//   Copyright notice:  This code shall not be replicated or used without //
//                      the permission of the author.                     //
//                                                                        //
/**************************************************************************/

/*

LIBRARY ROUTINES FOR ORTHOGONAL POLYNOMIAL CALCULUS AND INTERPOLATION

Spencer Sherwin
Center for Fluid Mechanics
Division of Applied Mathematics
Brown University
Providence
RI 02912,  USA

Based on splib.c by Einar Ronquist and Ron Henderson

Abbreviations
z    -   Set of collocation/quadrature points
w    -   Set of quadrature weights
d    -   Derivative operator
h    -   Lagrange Interpolant
i    -   Interpolation operator
g    -   Gauss
gr   -   Gauss-Radau
gl   -   Gauss-Lobatto
j    -   Jacobi
l    -   Legendre  (Jacobi with alpha = beta =  0.0)
c    -   Chebychev (Jacobi with alpha = beta = -0.5)
m    -   Arbritrary mesh

-----------------------------------------------------------------------
M A I N     R O U T I N E S
-----------------------------------------------------------------------

Points and Weights:

zwgj        Compute Gauss-Jacobi         points and weights
zwgrj       Compute Gauss-Radau-Jacobi   points and weights
zwglj       Compute Gauss-Lobatto-Jacobi points and weights

Derivative Operators:

dgj         Compute Gauss-Jacobi         derivative matrix
dgrj        Compute Gauss-Radau-Jacobi   derivative matrix
dglj        Compute Gauss-Lobatto-Jacobi derivative matrix

Lagrange Interpolants:

hgj         Compute Gauss-Jacobi         Lagrange interpolants
hgrj        Compute Gauss-Radau-Jacobi   Lagrange interpolants
hglj        Compute Gauss-Lobatto-Jacobi Lagrange interpolants

Interpolation Operators:

igjm        Compute interpolation operator gj->m
igrjm       Compute interpolation operator grj->m
igljm       Compute interpolation operator glj->m

-----------------------------------------------------------------------
L O C A L      R O U T I N E S
-----------------------------------------------------------------------

Polynomial Evaluation:

jacobf      Returns value of Jacobi polynomial for given points
jacobd      Returns derivative of Jacobi polynomial for given points
jacobz      Returns Jacobi polynomial zeros
jaczfd      Returns value and derivative of Jacobi poly. at point z

gammaf      Gamma function for integer values and halves

-----------------------------------------------------------------------
M A C R O S
-----------------------------------------------------------------------

Legendre  polynomial alpha = beta = 0
Chebychev polynomial alpha = beta = -0.5

Points and Weights:

zwgl        Compute Gauss-Legendre          points and weights
zwgrl       Compute Gauss-Radau-Legendre    points and weights
zwgll       Compute Gauss-Lobatto-Legendre  points and weights

zwgc        Compute Gauss-Chebychev         points and weights
zwgrc       Compute Gauss-Radau-Chebychev   points and weights
zwglc       Compute Gauss-Lobatto-Chebychev points and weights

Derivative Operators:

dgl         Compute Gauss-Legendre          derivative matrix
dgrl        Compute Gauss-Radau-Legendre    derivative matrix
dgll        Compute Gauss-Lobatto-Legendre  derivative matrix

dgc         Compute Gauss-Chebychev         derivative matrix
dgrc        Compute Gauss-Radau-Chebychev   derivative matrix
dglc        Compute Gauss-Lobatto-Chebychev derivative matrix

Lagrangian Interpolants:

hgl         Compute Gauss-Legendre          Lagrange interpolants
hgrl        Compute Gauss-Radau-Legendre    Lagrange interpolants
hgll        Compute Gauss-Lobatto-Legendre  Lagrange interpolants

hgc         Compute Gauss-Chebychev         Lagrange interpolants
hgrc        Compute Gauss-Radau-Chebychev   Lagrange interpolants
hglc        Compute Gauss-Lobatto-Chebychev Lagrange interpolants

Interpolation Operators:

iglm        Compute interpolation operator gl->m
igrlm       Compute interpolation operator grl->m
igllm       Compute interpolation operator gll->m

igcm        Compute interpolation operator gc->m
igrcm       Compute interpolation operator grc->m
iglcm       Compute interpolation operator glc->m

Polynomial functions

jacobf      Evaluate the Jacobi polynomial at vector of points
jacobd      Evaluate the derivative of Jacobi poly at vector of points

------------------------------------------------------------------------


Useful references:

[1] Gabor Szego: Orthogonal Polynomials, American Mathematical Society,
Providence, Rhode Island, 1939.
[2] Abramowitz & Stegun: Handbook of Mathematical Functions,
Dover, New York, 1972.
[3] Canuto, Hussaini, Quarteroni & Zang: Spectral Methods in Fluid
Dynamics, Springer-Verlag, 1988.
[4] Ghizzetti & Ossicini: Quadrature Formulae, Academic Press, 1970.


NOTES
-----
(1) All routines are double precision.
(2) All array subscripts start from zero, i.e. vector[0..N-1]
(3) Matrices should be allocated as true 2-dimensional arrays with
row and column indices starting from 0.
*/

#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <math.h>
#include <ElVis/Extensions/JacobiExtension/Polylib.h>

#define STOP  30
#define EPS   1.0e-12

//#define STOP  50
//#define EPS   1.0e-15

namespace ElVis
{
    namespace JacobiExtension
    {
        /* local functions */
        static void jacobz (int, double *, double , double);
        static void jaczfd (double, double *, double *, int, double, double);

        static double gammaF (double);

        /*----------------------------------------------------------------------
        zwgj() - Gauss-Jacobi Points and Weights

        Generate np Gauss Jacobi points (z) and weights (w) associated with
        the Jacobi polynomial P^{alpha,beta}_np (alpha >-1,beta >-1).

        Exact for polynomials of order 2np-1 or less
        -----------------------------------------------------------------------*/

        void zwgj (double *z, double *w, int np, double alpha, double beta){
            register int i;
            double fac, one = 1.0, two = 2.0, apb = alpha + beta;

            jacobz(np,z,alpha,beta);
            jacobd(np,z,w,np,alpha,beta);

            fac  = pow(two,apb + one)*gammaF(alpha + np + one)*gammaF(beta + np + one);
            fac /= gammaF(np + one)*gammaF(apb + np + one);

            for(i = 0; i < np; ++i) w[i] = fac/(w[i]*w[i]*(one-z[i]*z[i]));

            return;
        }

        /*----------------------------------------------------------------------
        zwgrj() - Gauss-Radau-Jacobi Points and Weights

        Generate np Gauss-Radau-Jacobi points (z) and weights (w) associated
        with the Jacobi polynomial P^(alpha,beta)_{np-1}.

        Exact for polynomials of order 2np-2 or less
        -----------------------------------------------------------------------*/

        void zwgrj(double *z, double *w, int np, double alpha, double beta){
            // recent FIX
            if(np == 1){
                z[0] = 0.0;
                w[0] = 2.0;
            }
            else{
                register int i;
                double fac, one = 1.0, two = 2.0, apb = alpha + beta;

                z[0] = -one;
                jacobz (np-1,z+1,alpha,beta+1);
                jacobf (np,z,w,np-1,alpha,beta);

                fac  = pow(two,apb)*gammaF(alpha + np)*gammaF(beta + np);
                fac /= gammaF(np)*(beta + np)*gammaF(apb + np + 1);

                for(i = 0; i < np; ++i) w[i] = fac*(1-z[i])/(w[i]*w[i]);
                w[0] *= (beta + one);
            }

            return;
        }


        /*---------------------------------------------------------------------
        Generate np Gauss-Lobatto-Jacobi points (z) and weights (w)
        associated with Jacobi polynomial P^{alpha,beta}_{np-2}
        (alpha>-1,beta>-1).

        Exact for polynomials of order 2np-3 or less
        ---------------------------------------------------------------------*/

        void zwglj(double *z, double *w, int np, double alpha, double beta){
            // recent FIX ??

            if( np == 1 ) {
                z[0] = 0.0;
                w[0] = 2.0;
            }
            else{
                register int i;
                double   fac, one = 1.0, apb = alpha + beta, two = 2.0;

                z[0]    = -one;
                z[np-1] =  one;
                jacobz (np-2,z + 1,alpha + one,beta + one);
                jacobf (np,z,w,np-1,alpha,beta);

                fac  = pow(two,apb + 1)*gammaF(alpha + np)*gammaF(beta + np);
                fac /= (np-1)*gammaF(np)*gammaF(alpha + beta + np + one);

                for(i = 0; i < np; ++i) w[i] = fac/(w[i]*w[i]);
                w[0]    *= (beta  + one);
                w[np-1] *= (alpha + one);
            }

            return;
        }


        /*---------------------------------------------------------------------
        dgj() - Compute the Derivative Matrix

        Compute the derivative matrix d and its transpose dt associated with
        the n_th order Lagrangian interpolants through the np Gauss-Jacobi
        points z.

        du
        --  = D   * u  evaluated at z = z
        dz     ij    j                   i

        NOTE: d and dt are both square matrices. alpha,beta > -1
        ---------------------------------------------------------------------*/

        void dgj(double **d, double **dt, double *z, int np,double alpha, double beta){

            double one = 1.0, two = 2.0;

            if (np <= 0){
                d[0][0] = dt[0][0] = 0.0;
            }
            else{
                register int i,j;
                double *pd;

                pd = (double *)malloc(np*sizeof(double));
                jacobd(np,z,pd,np,alpha,beta);

                for (i = 0; i < np; i++){
                    for (j = 0; j < np; j++){

                        if (i != j)
                            d[i][j] = pd[i]/(pd[j]*(z[i]-z[j]));
                        else
                            d[i][j] = (alpha - beta + (alpha + beta + two)*z[i])/
                            (two*(one - z[i]*z[i]));

                        dt[j][i] = d[i][j];
                    }
                }
                free(pd);
            }
            return;
        }


        /*---------------------------------------------------------------------
        dgrj() - Compute the Derivative Matrix

        Compute the derivative matrix d and its transpose dt associated with
        the n_th order Lagrangian interpolants through the np Gauss-Radau
        points z.

        du
        --  = D   * u  evaluated at z = z
        dz     ij    j                   i

        NOTE: d and dt are both square matrices. alpha,beta > -1
        ---------------------------------------------------------------------*/

        void dgrj(double **d, double **dt, double *z, int np,
            double alpha, double beta){

                if (np <= 0){
                    d[0][0] = dt[0][0] = 0.0;
                }
                else{
                    register int i, j;
                    double   one = 1.0, two = 2.0;
                    double   *pd;

                    pd  = (double *)malloc(np*sizeof(double));


                    pd[0] = pow(-one,np-1)*gammaF(np+beta+one);
                    pd[0] /= gammaF(np)*gammaF(beta+two);
                    jacobd(np-1,z+1,pd+1,np-1,alpha,beta+1);
                    for(i = 1; i < np; ++i) pd[i] *= (1+z[i]);

                    for (i = 0; i < np; i++)
                        for (j = 0; j < np; j++){
                            if (i != j)
                                d[i][j] = pd[i]/(pd[j]*(z[i]-z[j]));
                            else {
                                if(i == 0)
                                    d[i][j] = -(np + alpha + beta + one)*(np - one)/(two*(beta + two));
                                else
                                    d[i][j] = (alpha - beta + one + (alpha + beta + one)*z[i])/
                                    (two*(one - z[i]*z[i]));
                            }

                            dt[j][i] = d[i][j];
                        }
                        free(pd);
                }

                return;
        }

        /*---------------------------------------------------------------------
        dglj() - Compute the Derivative Matrix

        Compute the derivative matrix d and its transpose dt associated with
        the n_th order Lagrangian interpolants through the np Gauss-Lobatto
        points z.

        du
        --  = D   * u  evaluated at z = z
        dz     ij    j                   i

        NOTE: d and dt are both square matrices. alpha,beta > -1
        ---------------------------------------------------------------------*/

        void dglj(double **d, double **dt, double *z, int np,
            double alpha, double beta){

                if (np <= 0){
                    d[0][0] = dt[0][0] = 0.0;
                }
                else{
                    register int i, j;
                    double   one = 1.0, two = 2.0;
                    double   *pd;

                    pd  = (double *)malloc(np*sizeof(double));

                    pd[0]  = two*pow(-one,np)*gammaF(np + beta);
                    pd[0] /= gammaF(np - one)*gammaF(beta + two);
                    jacobd(np-2,z+1,pd+1,np-2,alpha+1,beta+1);
                    for(i = 1; i < np-1; ++i) pd[i] *= (one-z[i]*z[i]);
                    pd[np-1]  = -two*gammaF(np + alpha);
                    pd[np-1] /= gammaF(np - one)*gammaF(alpha + two);

                    for (i = 0; i < np; i++)
                        for (j = 0; j < np; j++){
                            if (i != j)
                                d[i][j] = pd[i]/(pd[j]*(z[i]-z[j]));
                            else {
                                if      (i == 0)
                                    d[i][j] = (alpha - (np - 1)*(np + alpha + beta))/(two*(beta+ two));
                                else if (i == np-1)
                                    d[i][j] =-(beta - (np - 1)*(np + alpha + beta))/(two*(alpha+ two));
                                else
                                    d[i][j] = (alpha - beta + (alpha + beta)*z[i])/
                                    (two*(one - z[i]*z[i]));
                            }

                            dt[j][i] = d[i][j];
                        }
                        free(pd);
                }

                return;
        }


        /*-------------------------------------------------------------
        Compute the value of the Lagrangian interpolant hglj through
        the np Gauss-Jacobi points zgj at the point z.
        -------------------------------------------------------------*/

        double hgj (int i, double z, double *zgj, int np, double alpha, double beta)
        {

            double zi, dz, p, pd, h;

            zi  = *(zgj+i);
            dz  = z - zi;
            if (fabs(dz) < EPS) return 1.0;

            jacobd(1, &zi, &pd, np, alpha, beta);
            jacobf(1, &z , &p , np, alpha, beta);
            h = p/(pd*dz);

            return h;
        }

        /*------------------------------------------------------------
        Compute the value of the Lagrangian interpolant hglj through
        the np Gauss-Radau-Jacobi points zgrj at the point z.
        ------------------------------------------------------------*/

        double hgrj (int i, double z, double *zgrj, int np, double alpha, double beta)
        {

            double zi, dz, p, pd, h;

            zi  = *(zgrj+i);
            dz  = z - zi;
            if (fabs(dz) < EPS) return 1.0;

            jacobf(1, &zi, &p , np-1, alpha, beta + 1);
            jacobd(1, &zi, &pd, np-1, alpha, beta + 1);
            h = (1.0 + zi)*pd + p;
            jacobf(1, &z, &p, np-1, alpha, beta + 1);
            h = (1.0 + z )*p/(h*dz);

            return h;
        }

        /*------------------------------------------------------------
        Compute the value of the Lagrangian interpolant hglj through
        the np Gauss-Lobatto-Jacobi points zjlj at the point z.
        ------------------------------------------------------------*/

        double hglj (int i, double z, double *zglj, int np, double alpha, double beta)
        {
            double one = 1., two = 2.;
            double zi, dz, p, pd, h;

            zi  = *(zglj+i);
            dz  = z - zi;
            if (fabs(dz) < EPS) return 1.0;

            jacobf(1, &zi, &p , np-2, alpha + one, beta + one);
            jacobd(1, &zi, &pd, np-2, alpha + one, beta + one);
            h = (one - zi*zi)*pd - two*zi*p;
            jacobf(1, &z, &p, np-2, alpha + one, beta + one);
            h = (one - z*z)*p/(h*dz);

            return h;
        }


        /*--------------------------------------------------------------------
        igjm() - Interpolation Operator GJ -> M

        Compute the one-dimensional interpolation operator (matrix) I12 for
        interpolating a function from a Gauss-Jacobi mesh (1) to another
        mesh M (2).
        ---------------------------------------------------------------------*/

        void igjm(double **im12,double *zgj, double *zm, int nz, int mz,
            double alpha, double beta){
                double zp;
                register int i, j;
#if 0
                if (nz == 1)
                    **im12 = 1.0;
                else
#endif
                    for (i = 0; i < mz; ++i) {
                        zp = zm[i];
                        for (j = 0; j < nz; ++j)
                            im12 [i][j] = hgj(j, zp, zgj, nz, alpha, beta);
                    }

                    return;
        }

        /*--------------------------------------------------------------------
        igrjm() - Interpolation Operator GRJ -> M

        Compute the one-dimensional interpolation operator (matrix) I12 for
        interpolating a function from a Gauss-Radau-Jacobi mesh (1) to
        another mesh M (2).
        ---------------------------------------------------------------------*/

        void igrjm(double **im12,double *zgrj, double *zm, int nz, int mz,
            double alpha, double beta){
                double zp;
                register int i, j;
#if 0
                if (nz == 1)
                    **im12 = 1.;
                else
#endif
                    for (i = 0; i < mz; i++) {
                        zp = zm[i];
                        for (j = 0; j < nz; j++)
                            im12 [i][j] = hgrj(j, zp, zgrj, nz, alpha, beta);
                    }

                    return;
        }

        /*--------------------------------------------------------------------
        igljm() - Interpolation Operator GRJ -> M

        Compute the one-dimensional interpolation operator (matrix) I12 and
        its transpose IT12 for interpolating a function from a Gauss-Labatto
        Jacobi mesh (1) to another mesh M (2).
        ----------------------------------------------------------------------*/

        void igljm(double **im12, double *zglj, double *zm, int nz, int mz,
            double alpha, double beta)
        {
            double zp;
            register int i, j;

#if 0
            if (nz == 1)
                **im12 = 1.;
            else
#endif
                for (i = 0; i < mz; i++) {
                    zp = zm[i];
                    for (j = 0; j < nz; j++)
                        im12 [i][j] = hglj(j, zp, zglj, nz, alpha, beta);
                }

                return;
        }

        /* -----------------------------------------------------------------
        jacobi() - jacobi polynomials

        Get a vector 'poly' of values of the n_th order Jacobi polynomial
        P^(alpha,beta)_n(z) alpha > -1, beta > -1 at the np points in z
        ----------------------------------------------------------------- */

        void jacobf(int np, double *z, double *poly, int n, double alpha, double beta){

            if(!np)
                return;

            register int i;
            double  one = 1.0, two = 2.0;

            if(n == 0)
                for(i = 0; i < np; ++i)
                    poly[i] = one;
            else if (n == 1)
                for(i = 0; i < np; ++i)
                    poly[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
            else{
                register int k;
                double   a1,a2,a3,a4;
                double   two = 2.0, apb = alpha + beta;
                double   *polyn1,*polyn2;

                polyn1 = (double *)malloc(np*sizeof(double));
                polyn2 = (double *)malloc(np*sizeof(double));

                for(i = 0; i < np; ++i){
                    polyn2[i] = one;
                    polyn1[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
                }

                for(k = 2; k <= n; ++k){
                    a1 =  two*k*(k + apb)*(two*k + apb - two);
                    a2 = (two*k + apb - one)*(alpha*alpha - beta*beta);
                    a3 = (two*k + apb - two)*(two*k + apb - one)*(two*k + apb);
                    a4 =  two*(k + alpha - one)*(k + beta - one)*(two*k + apb);

                    a2 /= a1;
                    a3 /= a1;
                    a4 /= a1;

                    for(i = 0; i < np; ++i){
                        poly  [i] = (a2 + a3*z[i])*polyn1[i] - a4*polyn2[i];
                        polyn2[i] = polyn1[i];
                        polyn1[i] = poly  [i];
                    }
                }
                free(polyn1); free(polyn2);
            }

            return;
        }

        void ortho_jacobf(int np, double *z, double *poly, int n, double alpha, double beta){

            if(!np)
                return;

            register int i;
            double  one = 1.0;
            double adjust = sqrt((2.0*n+1)/2.0);
            if(n == 0)
            {
                for(i = 0; i < np; ++i)
                {
                    poly[i] = adjust;
                }
            }
            else if (n == 1)
            {
                for(i = 0; i < np; ++i)
                {
                    //poly[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
                    poly[i] = z[i]*adjust;
                }
            }
            else{
                register int k;
                double   a1,a2,a3,a4;
                double   two = 2.0, apb = alpha + beta;
                double   *polyn1,*polyn2;

                polyn1 = (double *)malloc(np*sizeof(double));
                polyn2 = (double *)malloc(np*sizeof(double));

                for(i = 0; i < np; ++i){
                    polyn2[i] = one;
                    polyn1[i] = 0.5*(alpha - beta + (alpha + beta + two)*z[i]);
                    //  polyn2[i] = adjust;
                    //  polyn1[i] = z[i]*adjust;
                }

                for(k = 2; k <= n; ++k){
                    a1 =  two*k*(k + apb)*(two*k + apb - two);
                    a2 = (two*k + apb - one)*(alpha*alpha - beta*beta);
                    a3 = (two*k + apb - two)*(two*k + apb - one)*(two*k + apb);
                    a4 =  two*(k + alpha - one)*(k + beta - one)*(two*k + apb);

                    a2 /= a1;
                    a3 /= a1;
                    a4 /= a1;

                    for(i = 0; i < np; ++i){
                        poly  [i] = (a2 + a3*z[i])*polyn1[i] - a4*polyn2[i];
                        //poly[i] *= adjust;
                        polyn2[i] = polyn1[i];
                        polyn1[i] = poly  [i];
                    }
                }

                for(i = 0; i < np; ++i)
                {
                    poly[i] *= adjust;
                }
                free(polyn1); free(polyn2);
            }

            return;
        }

        /* ----------------------------------------------------------------
        jacobd() - derivative of jacobi polynomials - vector version

        Get a vector 'poly' of values of the derivative of the n_th order
        Jacobi polynomial P^(alpha,beta)_N(z) at the np points z.

        To do this we have used the relation

        d   alpha,beta   1                  alpha+1,beta+1
        -- P (z)       = -(alpha+beta+n+1) P  (z)
        dz  n            2                  n-1
        ----------------------------------------------------------------*/

        void jacobd(int np, double *z, double *polyd, int n, double alpha, double beta)
        {
            register int i;
            double one = 1.0;
            if(n == 0)
                for(i = 0; i < np; ++i) polyd[i] = 0.0;
            else{
                jacobf(np,z,polyd,n-1,alpha+one,beta+one);
                for(i = 0; i < np; ++i) polyd[i] *= 0.5*(alpha + beta + (double)n + one);
            }
            return;
        }

        /*----------------------------------------------------------
        jacobz() - zeros of the Jacobi polynomial

        Compute zeros z of the n_th order Jacobi  polynomial.
        alpha > -1, beta > -1
        -----------------------------------------------------------*/

        static void jaczfd (double, double *, double *, int , double, double);

        static void jacobz(int n, double *z, double alpha, double beta){

            if(!n)
                return;

            register int i,j,k;
            double   dth = M_PI/(2.0*(double)n);
            double   poly,pder,rlast=0.0;
            double   sum,delr,r;
            double one = 1.0, two = 2.0;

            for(k = 0; k < n; ++k){
                r = -cos((two*(double)k + one) * dth);
                if(k) r = 0.5*(r + rlast);

                for(j = 1; j < STOP; ++j){
                    jaczfd(r,&poly, &pder, n, alpha, beta);

                    for(i = 0, sum = 0.0; i < k; ++i) sum += one/(r - z[i]);

                    delr = -poly / (pder - sum * poly);
                    r   += delr;
                    if( fabs(delr) < EPS ) break;
                }
                z[k]  = r;
                rlast = r;
            }

            return;
        }

        /* -------------------------------------------------------------------
        This is a function to calculate the value of the n_th order jacobi
        polynomial and it's derivative at a point z for use with jacobz,
        hgj, hgrj and hgrl. (-1 < z < 1)
        ------------------------------------------------------------------*/
        static void jaczfd (double z, double *poly, double *polyd, int n,
            double alpha, double beta){
                double  one = 1.0, two = 2.0;

                if(n == 0){
                    *poly  = one;
                    *polyd = 0.0;
                }
                else if (n == 1){
                    *poly  = 0.5*(alpha - beta + (alpha + beta + two)*z);
                    *polyd = 0.5*(alpha + beta + two);
                }
                else{
                    register int k;
                    double   a1,a2,a3,a4;
                    double   apb = alpha + beta;
                    double   polyn1,polyn2;

                    polyn2 = one;
                    polyn1 = 0.5*(alpha - beta + (apb + two)*z);

                    for(k = 2; k <= n; ++k){
                        a1 =  two*k*(k + apb)*(two*k + apb - two);
                        a2 = (two*k + apb - one)*(alpha*alpha - beta*beta);
                        a3 = (two*k + apb - two)*(two*k + apb - one)*(two*k + apb);
                        a4 =  two*(k + alpha - one)*(k + beta - one)*(two*k + apb);

                        *poly  = ((a2 + a3*z)*polyn1 - a4*polyn2)/a1;
                        polyn2 = polyn1;
                        polyn1 = *poly;
                    }
                    polyn1 = polyn2;
                    *polyd  = n*(alpha - beta - (two*n + alpha + beta)*z)*poly[0];
                    *polyd += two*(n + alpha)*(n + beta)*polyn1;
                    *polyd /= (two*n + alpha + beta)*(one - z*z);
                }

                return;
        }

        /*-----------------------------------------------------------------------*
        * calculate the gamma function for integer values and halves            *
        * i.e.  gamma(n) = (n-1)!   gamma(n+1/2) = (n-1/2)*gamma(n-1/2)         *
        *                               where gamma(1/2) = sqrt(PI)             *
        *-----------------------------------------------------------------------*/

        static double gammaF(double x){
            double gamma = 1.0;

            if     (x == -0.5) gamma = -2.0*sqrt(M_PI);
            else if (!x) return gamma;
            else if ((x-(int)x) == 0.5){
                int n = (int) x;
                double tmp = x;

                gamma = sqrt(M_PI);
                while(n--){
                    tmp   -= 1.0;
                    gamma *= tmp;
                }
            }
            else if ((x-(int)x) == 0.0){
                int n = (int) x;
                double tmp = x;

                while(--n){
                    tmp   -= 1.0;
                    gamma *= tmp;
                }
            }
            else
                fprintf(stderr,"%lf is not of integer of half order\n",x);
            return gamma;
        }
    }
}

