////////////////////////////////////////////////////////////////////////////////
//
//  File: polylib.h
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

/*
*  LIBRARY ROUTINES FOR POLYNOMIAL CALCULUS AND INTERPOLATION
*/

#ifndef ELVIS_JACOBI_EXTENSION_ELVIS_HIGH_ORDER_ISOSURFACE_PLYLIB_H
#define ELVIS_JACOBI_EXTENSION_ELVIS_HIGH_ORDER_ISOSURFACE_PLYLIB_H

namespace ElVis
{
    namespace JacobiExtension
    {
#ifndef M_PI
#define M_PI  3.14159265358979323846
#endif

        /*-----------------------------------------------------------------------
        M A I N     R O U T I N E S
        -----------------------------------------------------------------------*/

        /* Points and weights */

        void   zwgj    (double *, double *, int , double , double);
        void   zwgrj   (double *, double *, int , double , double);
        void   zwglj   (double *, double *, int , double , double);

        /* Derivative operators */

        void   dgj     (double **, double **, double *, int, double, double);
        void   dgrj    (double **, double **, double *, int, double, double);
        void   dglj    (double **, double **, double *, int, double, double);

        /* Lagrangian interpolants */

        double hgj     (int, double, double *, int, double, double);
        double hgrj    (int, double, double *, int, double, double);
        double hglj    (int, double, double *, int, double, double);

        /* Interpolation operators */

        void   igjm  (double**, double*, double*, int, int, double, double);
        void   igrjm (double**, double*, double*, int, int, double, double);
        void   igljm (double**, double*, double*, int, int, double, double);

        /* Polynomial functions */

        void jacobf (int, double *, double *, int, double, double);
        void ortho_jacobf (int, double *, double *, int, double alpha = 0.0, double beta = 0.0);

        void jacobd (int, double *, double *, int, double, double);

        /*-----------------------------------------------------------------------
        M A C R O S
        -----------------------------------------------------------------------*/

        /* Points and weights */

#define  zwgl( z,w,np)   zwgj( z,w,np,0.0,0.0);
#define  zwgrl(z,w,np)   zwgrj(z,w,np,0.0,0.0);
#define  zwgll(z,w,np)   zwglj(z,w,np,0.0,0.0);

#define  zwgc( z,w,np)   zwgj( z,w,np,-0.5,-0.5);
#define  zwgrc(z,w,np)   zwgrj(z,w,np,-0.5,-0.5);
#define  zwglc(z,w,np)   zwglj(z,w,np,-0.5,-0.5);

        /* Derivative operators */

#define dgl( d,dt,z,np)  dgj( d,dt,z,np,0.0,0.0);
#define dgrl(d,dt,z,np)  dgrj(d,dt,z,np,0.0,0.0);
#define dgll(d,dt,z,np)  dglj(d,dt,z,np,0.0,0.0);

#define dgc( d,dt,z,np)  dgj( d,dt,z,np,-0.5,-0.5);
#define dgrc(d,dt,z,np)  dgrj(d,dt,z,np,-0.5,-0.5);
#define dglc(d,dt,z,np)  dglj(d,dt,z,np,-0.5,-0.5);

        /* Lagrangian interpolants */

#define hgl( i,z,zgj ,np)  hgj( i,z,zgj ,np,0.0,0.0);
#define hgrl(i,z,zgrj,np)  hgrj(i,z,zgrj,np,0.0,0.0);
#define hgll(i,z,zglj,np)  hglj(i,z,zglj,np,0.0,0.0);

#define hgc( i,z,zgj ,np)  hgj( i,z,zgj ,np,-0.5,-0.5);
#define hgrc(i,z,zgrj,np)  hgrj(i,z,zgrj,np,-0.5,-0.5);
#define hglc(i,z,zglj,np)  hglj(i,z,zglj,np,-0.5,-0.5);

        /* Interpolation operators */

#define iglm( im12,zgl ,zm,nz,mz) igjm( im12,zgl ,zm,nz,mz,0.0,0.0)
#define igrlm(im12,zgrl,zm,nz,mz) igrjm(im12,zgrl,zm,nz,mz,0.0,0.0)
#define igllm(im12,zgll,zm,nz,mz) igljm(im12,zgll,zm,nz,mz,0.0,0.0)

#define igcm( im12,zgl ,zm,nz,mz) igjm( im12,zgl ,zm,nz,mz,-0.5,-0.5)
#define igrcm(im12,zgrl,zm,nz,mz) igrjm(im12,zgrl,zm,nz,mz,-0.5,-0.5)
#define iglcm(im12,zgll,zm,nz,mz) igljm(im12,zgll,zm,nz,mz,-0.5,-0.5)
    }
}

#endif  //ELVIS_HIGH_ORDER_ISOSURFACE_PLYLIB_H










